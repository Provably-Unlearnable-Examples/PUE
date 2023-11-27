import torch
import copy
import collections
from collections import OrderedDict
import mlconfig
import shutil
import util
import os
import argparse
import datetime
import time
import dataset
import toolbox
import madrys
import math
import numpy as np
from scipy.stats import norm, binom 
from evaluator import Evaluator
from tqdm import tqdm
from trainer import Trainer
mlconfig.register(madrys.MadrysLoss)

# General Options
parser = argparse.ArgumentParser(description='Learnability Certification')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--surrogate_path', type=str, default='./test_exp/clean/norand/cifar10')
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--load_model', action='store_true', default=True)
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--use_subset', action='store_true', default=False)
# Cert Options
parser.add_argument('--q', default=0.7, type=float, help='Probability of the certification holds')
parser.add_argument('--sigma', default=0.1, type=float, help='STD of the smoothing noise')
parser.add_argument('--N', default=10, type=int, help='Number of models')
# Datasets Options
parser.add_argument('--train_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'ImageNetMini'])
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_type', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'ImageNetMini'])
parser.add_argument('--test_data_path', type=str, default='../datasets')
args = parser.parse_args()



def certify(acc_scores, q, eta, sigma):
    '''
        Certification algorithm
    '''
    c = 0.99
    q_upper = norm.cdf(norm.ppf(q) + (eta / sigma))
    logger.info(f'q_upper: {q_upper}')
    
    k_u_u = len(acc_scores) + 1
    k_u_l = math.ceil(q_upper * len(acc_scores))

    k_u_final = k_u_u
    for k_u in range(k_u_l, k_u_u):
        conf = binom.cdf(k_u-1, len(acc_scores), q_upper)
        logger.info(f"Current binomial CDF is {conf}")
        if conf > c:
            k_u_final = k_u - 1
            logger.info(f"Found k_u.")
            logger.info(f'k_u_final: {k_u_final}')
            break

    if k_u_final == k_u_u:
        return -1
    
    return k_u_final


def main():
    # Setup ENV
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed)
    data_loader = datasets_generator.getDataLoader(train_shuffle=True, test_drop_last=True)
    # debug code
    logger.info(f"\n train_dataset size {len(data_loader['train_dataset'].dataset)}, \n \
              test_dataset size {len(data_loader['test_dataset'].dataset)}. ")
    
    data, label = next(iter(data_loader['test_dataset']))
    print('data shape:', data.shape)
    print('label shape:', label.shape)

    model = config.model()
    model = model.to(device)
    logger.info("Base model param size = %fMB", util.count_parameters_in_MB(model))
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    logger.info("#"*20 + "Training Info Summary" + "#"*20 + '\n')
    evaluator = Evaluator(data_loader, logger, config)

    # Load model weights or train from scratch
    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        logger.info("File %s loaded!" % (checkpoint_path_file))
        evaluator.eval(0, model)
        base_acc = evaluator.acc_meters.avg*100
        payload = ('Eval Loss:%.4f \tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
    else:
        raise('Surrogate not found!')

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
        
    # Evaluate surrogate model on clear data
    acc_scores = []
    if args.test_data_type == "CIFAR10":
        classes = [0] * 10
    elif args.test_data_type == "CIFAR100" or args.test_data_type == "ImageNetMini":
        classes = [0] * 100
    else:
        raise("Not Implemented")
    total_evals = 0
    times = args.N
    model.eval()
    test_accuracy_avg = 0.0
    all_outputs = []

    logger.info('=' * 20 + 'Start Certification' + '=' * 20)
    for j in tqdm(range(times)):
        if j % 1 == 0:
            logger.info("=" * 10 + f"The {j}-th randomization." + "=" * 10)

        Noise = {}
        # Add noise
        for name, param in model.named_parameters():
            gaussian = torch.randn_like(param.data)
            Noise[name] = args.sigma * gaussian
            param.data = param.data + Noise[name]

        # Evaluate on the entire test set
        test_accuracy_local = 0.0
        for i, data in enumerate(data_loader['test_dataset'], 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            #print(f"Current batch size: {len(inputs)}")

            outputs = model(inputs)

            max_vals, max_indices = torch.max(outputs, 1)
            all_outputs.append(max_indices)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            test_accuracy_local += 100 * correct

            for val in max_indices:
                classes[val] += 1
                total_evals += 1

        # Compute accuracy
        test_accuracy_local /= len(data_loader['test_dataset'])
        if test_accuracy_local >= base_acc:
                logger.info("*" * 20 + "Find a better classifier after randomization"+ "*" * 20)
                logger.info(f"Current test accuracy: {test_accuracy_local}")
        test_accuracy_avg += test_accuracy_local
        acc_scores.append(test_accuracy_local)
        # remove the noise
        for name, param in model.named_parameters():
            param.data = param.data - Noise[name]

    test_accuracy_avg /= times

    logger.info("Average test accuracy: %.2f", test_accuracy_avg)
    logger.info("Classes counts: %s", classes)
    logger.info("Class disctribution: %s", 100 * (np.asarray(classes) / total_evals))
    #all_outputs = torch.stack(all_outputs)
    acc_scores.sort()
    acc_q = np.percentile(acc_scores, args.q*100)
    logger.info(f"Accuracy in the {args.q*100}-th percentile is: {acc_q}")

    etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for eta in etas:
        logger.info('#################' + 'Certifying' + '#################')
        logger.info("Eta: %s: ", eta)
        k = certify(acc_scores, args.q, eta, args.sigma)
        if k != -1:
            logger.info("Certified learnability score is %s", acc_scores[k])
        else:
            logger.info('Abstain')
    return


if __name__ == "__main__":
    # Set up Experiments
    if args.exp_name == '':
        args.exp_name = 'exp_' + datetime.datetime.now()
        
    exp_path = os.path.join(args.exp_name, args.version)
    log_file_path = os.path.join(exp_path, args.version)
    
    checkpoint_path = os.path.join(args.surrogate_path, args.version)
    checkpoint_path = os.path.join(checkpoint_path, 'checkpoints')
    checkpoint_path_file = os.path.join(checkpoint_path, args.version)
    util.build_dirs(exp_path)
    util.build_dirs(checkpoint_path)
    logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

    # CUDA Options
    #logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        #logger.info("GPU List: %s" % (device_list))
    else:
        device = torch.device('cpu')

    # Load Exp Configs
    config_file = os.path.join(args.config_path, args.version)+'.yaml'
    config = mlconfig.load(config_file)
    config.set_immutable()
    #for key in config:
    #    logger.info("%s: %s" % (key, config[key]))
    shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))
    
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Certification Cost %.2f Days \n" % cost
    logger.info(payload)
