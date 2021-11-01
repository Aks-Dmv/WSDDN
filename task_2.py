from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, get_box_data
from PIL import Image, ImageDraw

import sklearn.metrics


# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024

lr = 0.000001
momentum = 0.9
weight_decay = 0.0005
# ------------

USE_WANDB = True

if rand_seed is not None:
    np.random.seed(rand_seed)

# load datasets and create dataloaders

train_dataset = VOCDataset(image_size=512,top_n=300)
val_dataset = VOCDataset(split='test', image_size=512,top_n=300)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,   # batchsize is one for this implementation
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    sampler=None,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True)


# Create network and initialize
net = WSDDN(classes=train_dataset.CLASS_NAMES)
print(net)

if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()

for name, param in pret_net.items():
    print(name)
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue


# Move model to GPU and set train mode
# net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)
params = list(net.parameters())
optimizer = torch.optim.Adam(params[2:], lr=lr)




output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
disp_interval = 10
val_interval = 1000

if USE_WANDB:
    wandb.init(project="vlr2", reinit=True)

def test_net(model, val_loader=None, thresh=0.05):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """
    global cntr_valid, cntr_val_mean
    sigmoid = torch.nn.Sigmoid()
    for iter, data in enumerate(val_loader):

        # one batch = data for one image
        image           = data[0].cuda()
        target          = data[1].cuda()
        wgt             = data[2].cuda()
        rois            = data[3].cuda().float()
        gt_boxes        = torch.from_numpy(np.array(data[4])).cuda()
        gt_class_list   = torch.from_numpy(np.array(data[5])).cuda()

        #TODO: perform forward pass, compute cls_probs
        cls_probs = net(image, torch.ceil(rois*512), wgt, target)

        # print("inside test", cls_probs)
        # TODO: Iterate over each class (follow comments)
        # print("new test")
        valid = wgt.cpu().detach().numpy().flatten()
        gt = target.cpu().detach().numpy().flatten()
        cls_prob = torch.sum(cls_probs, 0).flatten()
        pred = torch.clamp(cls_prob, min=0.0, max=1.0).cpu().detach().numpy()
        if iter==0:
            concat_pred = pred
            concat_gt = gt
            concat_wgt = valid
        else:
            concat_pred = np.vstack((concat_pred, pred))
            concat_gt = np.vstack((concat_gt, gt))
            concat_wgt = np.vstack((concat_wgt, valid))
        
        for class_num in range(20):            
            # get valid rois and cls_scores based on thresh
            
            # use NMS to get boxes and scores
            boxes, scores = nms(rois[0], cls_probs[:, class_num])
            # print(class_num, " boxes ",boxes, "scores ",scores)
            if USE_WANDB and len(boxes)>0:
                boxes = [i.detach().cpu().tolist() for i in boxes]
                print(boxes)
                image = image.cpu()
                nums_boxes = range(len(boxes)) # placeholder for names of proposals
                wandb_img = wandb.Image(image, boxes={
                    "predictions": {
                        "box_data": get_box_data(nums_boxes, boxes)
                    },
                })
                wandb.log({"bboxes/proposals": wandb_img})
            
        
    AP = []
    for class_num in range(20): 
        gt_cls = concat_gt[:, class_num][concat_wgt[:, class_num] > 0].astype('float32')
        pred_cls = concat_pred[:, class_num][concat_wgt[:, class_num] > 0].astype('float32')
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

        if USE_WANDB:
            wandb.log({"val/ap_" +str(class_num): ap, "val/cntr":cntr_valid})
            cntr_valid+=1

        AP.append(ap)


    #TODO: visualize bounding box predictions when required
    #TODO: Calculate mAP on test set
    if USE_WANDB:
        wandb.log({"test/map": np.mean(AP), "test/cntr":cntr_val_mean})
        cntr_val_mean+=1
            
    return np.mean(AP)

cntr_val_mean = 0
cntr_valid = 0
cntr_train = 0
for _ in range(10):
    for iter, data in enumerate(train_loader):
        # print(iter)
        #TODO: get one batch and perform forward pass
        # one batch = data for one image
        image           = data[0].cuda()
        target          = data[1].cuda()
        wgt             = data[2].cuda()
        rois            = data[3].cuda().float()
        gt_boxes        = torch.from_numpy(np.array(data[4])).cuda()
        gt_class_list   = torch.from_numpy(np.array(data[5])).cuda()


        #TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
        # also convert inputs to cuda if training on GPU
        cls_probs = net(image, torch.ceil(rois*512), wgt, target)
        # print(cls_probs)





        # backward pass and update
        loss = net.loss  
        print(iter, loss)
        train_loss += loss.item()
        step_cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #TODO: evaluate the model every N iterations (N defined in handout)

        if iter%val_interval == 0 and iter != 0:
            net.eval()
            ap = test_net(net, val_loader)
            print("MAP ", ap)
            net.train()


        #TODO: Perform all visualizations here
        #The intervals for different things are defined in the handout
        if USE_WANDB and iter % 500 == 0:
            wandb.log({"train/loss": loss, "train/cntr":cntr_train})
            cntr_train+=1
