import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import numpy as np

from torchvision.ops import roi_pool, roi_align

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        #TODO: Define the WSDDN model
        self.pre_trained_alexnet = torchvision.models.alexnet(pretrained=True)
        self.features   = self.pre_trained_alexnet.features
        self.features[-1] = Identity()
        self.roi_pool   = lambda x, y: roi_pool(x, y, 6, 31/512.)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True)
        )

        self.score_fc   = nn.Linear(in_features=4096, out_features=20)
        self.bbox_fc    = nn.Linear(in_features=4096, out_features=20)

        
        # loss
        self.cross_entropy = None
        self.crierion = nn.BCELoss()


    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                wgt=None,
                gt_vec=None,
                ):
        

        #TODO: Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        x = self.features(image)
        # print(rois[0,:10])
        x = self.roi_pool(x, [rois[0]])
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        scores = self.score_fc(x)
        bboxes = self.bbox_fc(x)
        bboxes = F.softmax(bboxes, dim=0)
        scores = F.softmax(scores, dim=1)
        cls_prob = scores * bboxes

        if self.training:
            label_vec = gt_vec.view(1, self.n_classes)
            # print(cls_prob.shape, wgt.shape, label_vec.shape)
            self.cross_entropy = self.build_loss(cls_prob, label_vec, wgt)
        
        return cls_prob

    
    def build_loss(self, cls_prob, label_vec, wgt):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        #TODO: Compute the appropriate loss using the cls_prob that is the
        #output of forward()
        #Checkout forward() to see how it is called
        cls_prob = torch.sum(cls_prob, 0).view(-1, self.n_classes)
        cls_prob = torch.clamp(cls_prob, min=0.0, max=1.0)
        loss = self.crierion(cls_prob*wgt, label_vec*wgt)

        return loss
