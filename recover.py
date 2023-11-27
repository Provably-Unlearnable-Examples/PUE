# Recover poisoned classifier to clean one and record the weights diff

import copy
import argparse
import collections
import datetime
import os
import shutil
import time
import dataset
import mlconfig
import torch
import util
import madrys
import numpy as np
from evaluator import Evaluator
from tqdm import tqdm
from trainer import Trainer, l2_of_param_diff, project
import matplotlib.pyplot as plt
mlconfig.register(madrys.MadrysLoss)

# General Options
parser = argparse.ArgumentParser(description='Recover poisond classifier to clean one')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="train_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--surrogate_path', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
# Datasets Options
parser.add_argument('--train_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_path', type=str, default='../datasets')
parser.add_argument('--use_train_subset', action='store_true', default=False)
# Training Options
parser.add_argument('--eta', default=1.0, type=float, help='parameter l2 radius')
parser.add_argument('--greedy', action='store_true', default=False, help='Use Gaussian greedy search')
args = parser.parse_args()


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
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


def random_greedy(model, poison_model, data_loader, eta):
    poison_model.eval()
    model.eval()
    origin_param = {}
    for name, param in poison_model.named_parameters():
            origin_param[name] = param.data
    
    best_acc = 0
    for i in range(1000):
        # Add noise
        for name, param in model.named_parameters():
            gaussian = torch.randn_like(param.data)
            param.data = param.data + 1.25 * gaussian
        project(model, poison_model, origin_param, eta)

        # Evaluate on the entire test set
        test_accuracy_local = 0.0
        for i, data in enumerate(data_loader['test_dataset'], 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            #print(f"Current batch size: {len(inputs)}")

            outputs = model(inputs)
            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            test_accuracy_local += 100 * correct

        # Compute accuracy
        test_accuracy_local /= len(data_loader['test_dataset'])
        if test_accuracy_local >= best_acc:
                best_acc = test_accuracy_local
                logger.info("*" * 20 + "Find a better classifier by greedy search"+ "*" * 20)
                logger.info(f"Current best test accuracy: {test_accuracy_local}")

        # remove the noise
        for name, param in model.named_parameters():
            param.data = origin_param[name]
    
    l2_diff = l2_of_param_diff(model, poison_model)
    logger.info(f"l2 diff of param change is {l2_diff}, Best acc is {best_acc}.") 
    return



def train(starting_epoch, model, poison_model, optimizer, scheduler, trainer, evaluator, ENV, eta):
    poison_model.eval()
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train_recovery(epoch, model, optimizer, poison_model, eta)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        evaluator.eval(epoch, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100
        l2_diff = l2_of_param_diff(poison_model, model).cpu().numpy()
        logger.info(f"Epoch {epoch}: l2_diff {l2_diff}, acc: {ENV['curren_acc']}")
        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()
    return


def main():
    model = config.model().to(device)
    datasets_generator = config.dataset(train_data_type=args.train_data_type,
                                        train_data_path=args.train_data_path,
                                        test_data_type=args.test_data_type,
                                        test_data_path=args.test_data_path,
                                        train_batch_size=args.train_batch_size,
                                        eval_batch_size=args.eval_batch_size,
                                        num_of_workers=args.num_of_workers,
                                        seed=args.seed)
    # logger.info('Training Dataset: %s' % str(datasets_generator.datasets['train_dataset']))
    # logger.info('Test Dataset: %s' % str(datasets_generator.datasets['test_dataset']))

    
    if args.use_train_subset:
        train_target = 'train_subset'
        data_loader = datasets_generator._split_validation_set(train_portion=0.2, train_shuffle=True, train_drop_last=True)
    else:
        train_target = 'recovery_dataset'
        data_loader = datasets_generator.getRecoveryDataLoader(train_shuffle=True, train_drop_last=True, test_drop_last=False)

    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    logger.info(f"Dataloader info: {str(data_loader[train_target].dataset)}")

    optimizer = config.optimizer(model.parameters())
    print(optimizer.param_groups[0]['lr'])
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()
    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)

    starting_epoch = 0
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': [],
           'cm_history': [],}

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        #reset optimizer and scheduler
        optimizer = config.optimizer(model.parameters())
        scheduler = config.scheduler(optimizer)
        logger.info("File %s loaded!" % (checkpoint_path_file))
        poison_model = copy.deepcopy(model)
        logger.info("Model %s copied!" % (checkpoint_path_file))
        evaluator.eval(0, model)
        payload = ('Loaded model Eval Loss:%.4f \tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        evaluator.eval(0, poison_model)
        payload = ('Copied model Eval Loss:%.4f \tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)

    print(optimizer.param_groups[0]['lr'])

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    if args.greedy:
        logger.info("="*20 + "Start random search" + "="*20)
        random_greedy(model, poison_model, data_loader, args.eta)
    else:
        logger.info("="*20 + "Start pSGD attack" + "="*20)
        train(0, model, poison_model, optimizer, scheduler, trainer, evaluator, ENV, args.eta)
    

if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)