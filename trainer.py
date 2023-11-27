import time
import models
import torch
import util
from torchensemble import BaggingClassifier
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

CIFAR10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
file = open("../datasets/imagenet-100/Labels_100.json")
ImageNet_Mini_labels = json.load(file)
#### The label matching problem should be fixed !!!
ImageNet_Mini_labels = list(zip(range(100), list(ImageNet_Mini_labels.values())))

def l2_of_param_diff(model_A, model_B):
    A_params = []
    B_params = []
    for _, param in model_A.named_parameters():
        A_params.append(param.data.reshape(-1))
    for _, param in model_B.named_parameters():
        B_params.append(param.data.reshape(-1))
    diff = torch.cat(A_params) - torch.cat(B_params)
    return torch.linalg.norm(diff, ord=2)


def project(model, poison_model, origin, eta):
    l2_diff = l2_of_param_diff(model, poison_model)
    #print("l2 diff is", l2_diff)
    for name, param in model.named_parameters():
        upsilon = (param.data - origin[name]) * eta/l2_diff
        param.data = origin[name] + upsilon
    return

class Trainer():
    def __init__(self, criterion, data_loader, logger, config, global_step=0, data_type='CIFAR10', target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        if data_type == 'ImageNetMini':
            self.labelmap = ImageNet_Mini_labels
        elif data_type == 'CIFAR10':
            self.labelmap = CIFAR10_labels
        print(f'Training set used for the trainer is: {self.target}')

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train(self, epoch, model, optimizer, robust_noise=0, robust_noise_step=0, avgtimes=0, random_noise=None, pue=False):
        model.train()
        pbar = tqdm(enumerate(self.data_loader[self.target]))
        for i, (images, labels) in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if random_noise is not None:
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
            start = time.time()
            if pue:
                log_payload = self.train_batch_robust(images, labels, model, optimizer, robust_noise, robust_noise_step, avgtimes, pbar)
            else:
                log_payload = self.train_batch(images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def train_batch(self, images, labels, model, optimizer):
        model.zero_grad()
        optimizer.zero_grad()
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            logits = model(images)
            loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
        if isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            _, labels = torch.max(labels.data, 1)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload
    
    def train_batch_robust(self, images, labels, model, optimizer, robust_noise, robust_noise_step, avgtimes, outer_pbar):
        model.zero_grad()
        plt.imshow(images[0].detach().cpu().numpy().transpose(1,2,0))
        plt.savefig(f"sample_training_image.png")
        times = int(robust_noise / robust_noise_step) + 1
        in_times = avgtimes
        for j in range(times):
            outer_pbar.set_postfix({'Inner step': j})
            optimizer.zero_grad()
            for k in range(in_times):
                Noise = {}
                # Add noise
                for name, param in model.named_parameters():
                    gaussian = torch.randn_like(param.data) * 1
                    Noise[name] = robust_noise_step * j * gaussian
                    param.data = param.data + Noise[name]

                # get loss
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
                    logits = model(images)
                    class_loss =  self.criterion(logits, labels)
                else:
                    logits, loss = self.criterion(model, images, labels, optimizer)
                if isinstance(self.criterion, models.CutMixCrossEntropyLoss):
                    _, labels = torch.max(labels.data, 1)
                loss = class_loss / (times * in_times)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

                # remove the noise
                for name, param in model.named_parameters():
                    param.data = param.data - Noise[name]
            optimizer.step()
        
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload
    
    def train_recovery(self, epoch, model, optimizer, poison_model, eta):
        model.train()
        origin_param = {}
        for name, param in poison_model.named_parameters():
                origin_param[name] = param.data
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, poison_model, origin_param, eta, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step
    
    def train_recovery_batch(self, images, labels, model, poison_model, origin_param, eta, optimizer):
        model.zero_grad()
        optimizer.zero_grad()
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            logits = model(images)
            loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
        if isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            _, labels = torch.max(labels.data, 1)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        project(model, poison_model, origin_param, eta)
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload
