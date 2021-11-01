import argparse
import os
import shutil
import time
import sys
import sklearn
import sklearn.metrics

import torch
torch.cuda.init()
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from AlexNet import *
from voc_dataset import *
from utils import *

import wandb
USE_WANDB = True # use flags, wandb is not convenient for debugging


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0
cntr_train = 0
cntr_val = 0

def main():
    global args, best_prec1, cntr_train, cntr_val
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model = torch.nn.DataParallel(model)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # you can also use PlateauLR scheduler, which usually works well
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code
    
    #TODO: Create Datasets and Dataloaders using VOCDataset - Ensure that the sizes are as required
    # Also ensure that data directories are correct - the ones use for testing by TAs might be different
    # Resize the images to 512x512
    train_dataset = VOCDataset(image_size=512)
    val_dataset = VOCDataset(split='test', image_size=512)
    
    def collate_fn(batch):
        return tuple(zip(*batch))



    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True, collate_fn=collate_fn)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    
    
    # TODO: Create loggers for wandb - ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="vlr2", reinit=True)


    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch)
        # training_scheduler.step(loss)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)




#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    global cntr_train
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO: Get inputs from the data dict


        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output such as clamping
        # TODO: Compute loss using ``criterion``
        img_input = torch.stack(data[0], dim=0).cuda()
        target = torch.stack(data[1], dim=0).cuda()
        wgt = torch.stack(data[2], dim=0).cuda()
        
        


        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output such as clamping
        # TODO: Compute loss using ``criterion``
        optimizer.zero_grad()
        output_heatmap = model(img_input)
        
        if args.arch == 'localizer_alexnet':
            max_pool_k = output_heatmap.shape[2]
            maxPool = nn.MaxPool2d(kernel_size=max_pool_k)
            output = maxPool(output_heatmap)
        elif args.arch == 'localizer_alexnet_robust':
            max_pool_k = output_heatmap[0].shape[2]
            maxPool = nn.MaxPool2d(kernel_size=max_pool_k)
            output = maxPool(output_heatmap[0])
            
            max_pool_k1 = output_heatmap[1].shape[2]
            maxPool1 = nn.MaxPool2d(kernel_size=max_pool_k1)
            output_1 = maxPool1(output_heatmap[1])
            
            max_pool_k2 = output_heatmap[2].shape[2]
            maxPool2 = nn.MaxPool2d(kernel_size=max_pool_k2)
            output_2 = maxPool2(output_heatmap[2])
            
            output = output*0.333 + output_1*0.333 + output_2*0.333
            
        
        output = output.view(output.shape[0], output.shape[1])
        
        loss = criterion(output*wgt, target*wgt)
        


        # measure metrics and record loss
        sigmoid = nn.Sigmoid()
        m1 = metric1(sigmoid(output), target, wgt)
        m2 = metric2(sigmoid(output), target, wgt)
        losses.update(loss.item(), img_input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # TODO:
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize/log things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        if USE_WANDB and i % args.print_freq == 0:
            wandb.log({"train/loss": loss, "train/cntr":cntr_train})
            wandb.log({"train/m1": m1, "train/cntr":cntr_train})
            wandb.log({"train/m2": m2, "train/cntr":cntr_train})
            cntr_train+=1




        # End of train()
    return loss.detach()


def validate(val_loader, model, criterion, epoch = 0):
    global cntr_val
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO: Get inputs from the data dict
        img_input = torch.stack(data[0], dim=0).cuda()
        target = torch.stack(data[1], dim=0).cuda()
        wgt = torch.stack(data[2], dim=0).cuda()
        


        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        output_heatmap = model(img_input)
        
        if args.arch == 'localizer_alexnet':
            max_pool_k = output_heatmap.shape[2]
            maxPool = nn.MaxPool2d(kernel_size=max_pool_k)
            output = maxPool(output_heatmap)
        elif args.arch == 'localizer_alexnet_robust':
            max_pool_k = output_heatmap[0].shape[2]
            maxPool = nn.MaxPool2d(kernel_size=max_pool_k)
            output = maxPool(output_heatmap[0])
            
            max_pool_k1 = output_heatmap[1].shape[2]
            maxPool1 = nn.MaxPool2d(kernel_size=max_pool_k1)
            output_1 = maxPool1(output_heatmap[1])
            
            max_pool_k2 = output_heatmap[2].shape[2]
            maxPool2 = nn.MaxPool2d(kernel_size=max_pool_k2)
            output_2 = maxPool2(output_heatmap[2])
            
            output = output*0.333 + output_1*0.333 + output_2*0.333
        
        output = output.view(output.shape[0], output.shape[1])
        
        loss = criterion(output*wgt, target*wgt)
        

        sigmoid = nn.Sigmoid()
        # measure metrics and record loss
        m1 = metric1(sigmoid(output), target, wgt)
        m2 = metric2(sigmoid(output), target, wgt)
        losses.update(loss.item(), img_input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        if USE_WANDB:
            if i % args.print_freq == 0:
                wandb.log({"val/loss": loss, "val/cntr":cntr_val})
                wandb.log({"val/m1": m1, "val/cntr":cntr_val})
                wandb.log({"val/m2": m2, "val/cntr":cntr_val})
                cntr_val+=1
            if i<5 and epoch%14==0:
                gt_np_img = img_input[0].detach().cpu().numpy().mean(axis=0)
                wandb.log({'heatmaps/epoch_{}_gt_img_{}'.format(epoch, i): wandb.Image(gt_np_img)})
                
                
                
                weighted_target = (target[0] * wgt[0]).detach().cpu().numpy()
                heat_i = 0
                resize512 = transforms.Resize((512, 512))
                for class_i in range(20):
                    print(weighted_target[class_i])
                    if weighted_target[class_i]==1:
                        target_gt = class_i
                    else:
                        continue
                    if args.arch == 'localizer_alexnet':
                        print("output heatmap shape ", output_heatmap.shape)
                        print(torch.sum(torch.isnan(output_heatmap[0,target_gt]).type(torch.uint8)))
                        out_heat = resize512(output_heatmap[0,target_gt][None,:,:])
                        selected_heatmap = out_heat.detach().cpu()
                        # selected_heatmap = selected_heatmap[None,:,:]
                    elif args.arch == 'localizer_alexnet_robust':
                        print("output heatmap shape ", output_heatmap[0].shape, output_heatmap[1].shape, output_heatmap[2].shape)
                        # print(torch.sum(torch.isnan(output_heatmap[0][0,target_gt]).type(torch.uint8)))
                        out_heat = resize512(output_heatmap[0][0,target_gt][None,:,:]) * 0.333
                        out_heat1 = resize512(output_heatmap[1][0,target_gt][None,:,:]) * 0.333
                        out_heat2 = resize512(output_heatmap[2][0,target_gt][None,:,:]) * 0.333
                        
                        selected_heatmap = out_heat + out_heat1 + out_heat2
                        selected_heatmap = selected_heatmap.detach().cpu()
                    
                    print("target gt", target_gt)
                    
                    
                    selected_heatmap = resize512(selected_heatmap)
                    selected_heatmap = torch.permute(selected_heatmap, (1,2,0)).numpy()

                    print(selected_heatmap.min())
                    print(selected_heatmap.max())

                    wandb.log({'heatmaps/epoch_{}_img_{}_heatmap_{}'.format(epoch, i, target_gt): wandb.Image(selected_heatmap)})


    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metric1(pred, gt, valid):
    # TODO: Ignore for now - proceed till instructed
    pred = torch.sigmoid(pred).cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    valid = valid.cpu().detach().numpy()
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        if np.all(gt_cls==0):
            if np.all(pred_cls<0.5):
                ap=1.
            else:
                ap=0.
        else:
            # As per PhilK. code:
            # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
            pred_cls -= 1e-5 * gt_cls
            ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls)
        AP.append(ap)
    
    
    return np.mean(AP)


def metric2(pred, gt, valid):
    #TODO: Ignore for now - proceed till instructed
    pred = torch.sigmoid(pred).cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    valid = valid.cpu().detach().numpy()
    nclasses = gt.shape[1]
    M2 = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        if np.all(gt_cls==0):
            if np.all(pred_cls<0.5):
                rec=1.
            else:
                rec=0.
        else:
            # As per PhilK. code:
            # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
            pred_cls -= 1e-5 * gt_cls
            # print(gt_cls)
            # print(pred_cls)
            rec = sklearn.metrics.recall_score(gt_cls, pred_cls>0.5, average='binary')
        M2.append(rec)
    
    
    return np.mean(M2)


if __name__ == '__main__':
    main()
