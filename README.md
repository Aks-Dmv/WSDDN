# Weakly Supervised Object Localization

This repository was made in partial fulfillment for the course [Visual Learning and Recognition (16-824) Fall 2021](https://visual-learning.cs.cmu.edu/), which I took at CMU. 

This repository contains code that trains object detectors in a *weakly supervised* setting, which means you're going to train object detectors without bounding box annotations.

We use [PyTorch](pytorch.org) to create our models and [Weights and Biases](https://wandb.ai/site) for visualizations and logging. We implemented a slightly simplified version of the following papers:

1. Oquab, Maxime, et al. "*Is object localization for free?-weakly-supervised learning with convolutional neural networks.*" Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [Link](https://www.di.ens.fr/~josef/publications/Oquab15.pdf)
2. Bilen, Hakan, and Andrea Vedaldi. "*Weakly supervised deep detection networks*." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. [Link](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16/bilen16.pdf)

We used the [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) dataset for training, to be consistent with the results of [WSDDN](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16/bilen16.pdf). The Pascal VOC dataset comes with bounding box annotations, however, we do not use bounding box annotations as we are in the weakly supervised setting. 


## Software Setup

The following Python libraries are required for running the repository:

1. PyTorch
2. Weights and Biases
3. SKLearn
4. Pillow (PIL)
5. Weights and Biases (wandb)


### Data setup
To download the image dataset you can use the code below. The data below should be stored in the folder `data`.
```bash
$ cd data
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ tar xf VOCtrainval_06-Nov-2007.tar
$ # Also download the test data
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar xf VOCtest_06-Nov-2007.tar
$ cd VOCdevkit/VOC2007/
$ export DATA_DIR=$(pwd)
```
WSDDN [2] requires bounding box proposals from either Selective Search, Edge Boxes or any other similar method. We can download these proposals from the follwoing link. You need to put these proposals in the `data` folder too. The code below can be used to download these models.
	
```bash
# You can run these commands to populate the data directory
$ # First, cd to the main code folder
$ cd data/VOCdevkit/VOC2007/
$ # Download the selective search data
$ wget https://www.cs.cmu.edu/~spurushw/files/selective_search_data.tar && tar xf selective_search_data.tar
```


## Task 1: Is Object Localization Free?
A good way to dive into using PyTorch is training a simple classification model on ImageNet. 
We won't be doing that to save the rainforest (and AWS credits) but you should take a look at the code [here](https://github.com/pytorch/examples/blob/master/imagenet/main.py). We will be following the same structure.

The code for the model is in `AlexNet.py`. In the code, you need to fill in the parts that say "TODO" (read the questions before you start filling in code). 
We need to define our model in one of the "TODO" parts. We are going to call this ``LocalizerAlexNet``. I've written a skeleton structure in `AlexNet.py`. You can look at the AlexNet example of PyTorch. For simplicity and speed, we won't be copying the FC layers to our model. We want the model to look like this:
```text
LocalizerAlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
  )
  (classifier): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (3): ReLU(inplace)
    (4): Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

#### Q 1.1 Fill in each of the TODO parts except for the functions ``metric1``, ``metric2`` and ``LocalizerAlexNetRobust``. In the report, for each of the TODO, describe the functionality of that part. The output of the above model has some spatial resolution. Make sure you read paper [1] and understand how to go from the output to an image level prediction (max-pool). (Hint: This part will be implemented in ``train()`` and ``validate()``.

#### Q 1.2 What is the output resolution of the model?

#### Plotting using Weights and Biases
Logging to [Weights and Biases](https://docs.wandb.ai/quickstart), also known as `wandb` is quite easy and super useful. You can use this to keep track of experiment hyperparameters and metrics such as loss/accuracy.
```python
import wandb
wandb.init(project="vlr-hw2")
# logging the loss
wandb.log({'epoch': epoch, 'loss': loss})
```
You can also use it to save models, perform hyperparameter tuning, and other cool stuff.

When you're logging to WandB, make sure you use good tag names. For example, for all training plots you can use ``train/loss``, ``train/metric1``, etc and for validation ``validation/metric1``, etc.

#### Q 1.3 Initialize the model from ImageNet (till the conv5 layer). Initialize the rest of layers with Xavier initialization and train the model using batchsize=32, learning rate=0.01, epochs=2 (Yes, only 2 epochs for now).(Hint: also try lr=0.1 - best value varies with implementation of loss)
- Use wandb to plot the training loss curve.
- Use wandb to plot images and the rescaled heatmaps for only the GT classes for 2 batches (1 images in each batch) in every epoch (uniformly spaced in iterations).

#### Q 1.4 In the first few iterations, you should observe a steep drop in the loss value. Why does this happen? (Hint: Think about the labels associated with each image).

#### Q 1.5 We will log two metrics during training to see if our model is improving progressively with iterations. The first metric is a standard metric for multi-label classification. Do you remember what this is? Write the code for this metric in the TODO block for ``metric1`` (make sure you handle all the boundary cases). However, ``metric1`` is to some extent not robust to the issue we identified in Q1.4. The second metric, Recall, is more tuned to this dataset. Even though there is a steep drop in loss in the first few iterations ``metric2`` should remain almost constant. Implement it in the TODO block for ``metric2``. (Make any assumptions needed - like thresholds).

### We're ready to train now!

#### Q 1.6 Initialize the model from ImageNet (till the conv5 layer), initialize the rest of layers with Xavier initialization and train the model using batchsize=32, learning rate=0.01, epochs=30. Evaluate every 2 epochs. (Hint: also try lr=0.1 - best value varies with implementation of loss) \[Expected training time: 45mins-75mins].
- IMPORTANT: FOR ALL EXPERIMENTS FROM HERE - ENSURE THAT THE SAME IMAGES ARE PLOTTED ACROSS EXPERIMENTS BY KEEPING THE SAMPLED BATCHES IN THE SAME ORDER. THIS CAN BE DONE BY FIXING THE RANDOM SEEDS BEFORE CREATING DATALOADERS.
- Use wandb to plot the training loss curve, training ``metric1``, training ``metric2``
- Use wandb to plot the mean validation ``metric1`` and mean validation ``metric2`` for every 2 epochs.
- Use wandb to plot images and the rescaled heatmaps for only the GT classes for 2 batches (1 images in each batch) at the end of the 1st, 15th, and last(30th) epoch. 

- At the end of training, use wandb to plot 3 randomly chosen images and corresponding heatmaps (similar to above) from the validation set.
- In your report, mention the training loss, training and validation ``metric1`` and ``metric2`` achieved at the end of training. 


#### Q 1.7 In the heatmap visualizations you observe that there are usually peaks on salient features of the objects but not on the entire objects. How can you fix this in the architecture of the model? (Hint: during training the max-pool operation picks the most salient location). Implement this new model in ``LocalizerAlexNetRobust`` and also implement the corresponding ``localizer_alexnet_robust()``. Train the model using batchsize=32, learning rate=0.01, epochs=45. Evaluate every 2 epochs.(Hint: also try lr=0.1 - best value varies with implementation of loss)
- For this question only visualize images and heatmaps using wandb at similar intervals as before (ensure that the same images are plotted). 
- You don't have to plot the rest of the quantities that you did for previous questions (if you haven't put flags to turn off logging the other quantities, it's okay to log them too - just don't add them to the report).
- At the end of training, use wandb to plot 3 randomly chosen images (same images as Q1.6) and corresponding heatmaps from the validation set.
- Report the training loss, training and validation ``metric1`` and ``metric2`` achieved at the end of training. 


## Task 2: Weakly Supervised Deep Detection Networks

First, make sure you understand the WSDDN model. 

The main script for training is ``task_2.py``.  Read all the comments to understand what each part does. There are a few major components that you need to work on:

- The network architecture and functionality ``WSDDN``
- Writing the traning loop and including visualizations for metrics
- The function `test_net()` in `task_2.py`, which will log metrics on the test set

Tip for Task 2: conda install tmux, and train in a tmux session. That way you can detach, close your laptop (don't stop your ec2 instance!), and go enjoy a cookie while you wait.

#### Q2.1 In ``wsddn.py``, you need to complete the  ``__init__, forward`` and `` build_loss`` functions. 
The `__init__()` function will be used to define the model. You can define 3 parts for the model
1. The feature extractor
2. A ROI-pool layer (use torchvision.ops)
3. A classifier layer, as defined in the WSDDN paper.

The `forward()` function will essentially do the following:
1. Extract features from a given image (notice that batch size is 1 here).
2. Use the regions proposed from Selective Search, perform ROI Pooling. There are 2 caveats here - ensure that the proposals are now absolute values of pixels in the input image, and ensure that the scale factor passed into the ROI Pooling layer works correctly for the given image and features [ref](https://discuss.pytorch.org/t/spatial-scale-in-torchvision-ops-roi-pool/59270).
3. For each image, ROI Pooling gives us a feature map for the proposed regions. Pass these features into the classifier subnetwork. Here, you can think of batch size being the number of region proposals for each image.
4. Combine the classifier outputs (for boxes and classes), which will give you a tensor of shape (N_boxes x 20). Return this.

The `build_loss()` function now computes classification loss, which can be accessed in the training loop.


#### Q2.2 In ``task_2.py`` you will first need to write the training loop.
This involves creating the dataset, calling the dataloaders, etc. and then finally starting the training loop with the forward and backward passes. Some of this functionality has already been implemented for you. Ideally, use the hyperparameters given in the code. You don't need to implement the visualizations yet.
Use `top_n=300`, but feel free to increase it as well.

#### Q2.3 In ``task_2.py``, you now need to write a function to test your model, and visualize its predictions.
1. Write a test loop similar to the training loop, and calculate mAP as well as class-wise AP's.

At this point, we have our model giving us (N_boxes x 20) scores. We can interpret this as follows - for each of the 20 classes, there are `N` boxes, which have confidence scores for that particular class. Now, we need to perform Non-Max Suppression for the bbox scores corresponding to each class.
- In `utils.py`, write the NMS function. NMS depends on the calculation of Intersection Over Union (IoU), which you can either implement as a separate function, or vectorize within the NMS function itself. Use an IoU threshold of 0.3.
- Use NMS with a confidence threshold of 0.05 (basically consider only confidence above this value) to remove unimportant bounding boxes for each class.
- In the test code, iterate over indices for each class. For each class, visualize the NMSed bounding boxes with the class names and the confidence scores. You can use wandb for this, but if using ImageDraw or something else is more convenient, feel free to use that instead.


#### Q2.4 In ``task_2.py``, there are places for you perform visualization (search for TODO). You need to perform the appropriate visualizations mentioned here:
- Plot the loss every 500 iterations using wandb.
- Use wandb to plot mAP on the *test* set every epoch.
- Plot the class-wise APs at every epoch.
- Plot bounding boxes on 10 random images at the end of the first epoch, and at the end of the last epoch. (You can visualize for more images, and choose whichever ones you feel represent the learning of the network the best. It's also interesting to see the kind of mistakes the network makes as it is learning, and also after it has learned a little bit!)

#### Q2.5 Train the model using the hyperparameters provided for 5-6 epochs.
The expected values for the metrics at the end of training are:
- Train Loss: ~1.0
- Test  mAP : ~0.13

Include all the code and images/logs after training.
Report the final class-wise AP on the test set and the mAP.



# Submission Checklist 
## Report

### Task 0
- [ ] Answer Q0.1, Q0.2
- [ ] wandb screenshot for Q0.3
- [ ] wandb screenshot for Q0.4
### Task 1
- [ ] Q1.1 describe functionality of the completed TODO blocks
- [ ] Answer Q1.2
- [ ] Answer Q1.4
- [ ] Answer Q1.5 and describe functionality of the completed TODO blocks
- [ ] Q1.6
	- [ ] Add screenshot of metric1, metric2 on the training set
	- [ ] Add screenshot of metric1, metric2 on the validation set
	- [ ] Screenshot of wandb showing images and heat maps for the first logged epoch
	- [ ] Screenshot of wandb showing images and heat maps for the last logged epoch
	- [ ] wandb screenshot for 3 randomly chosen validation images and heat maps
	- [ ] Report training loss, validation metric1, validation metric2 at the end of training

- [ ] Q1.7 
	- [ ] Screenshot of wandb showing images and heat maps for the first logged epoch \*for Q1.6 and Q1.7 show image and heatmap side-by-side\*.
	- [ ] Screenshot of wandb showing images and heat maps for the last logged epoch \*for Q1.6 and Q1.7 show image and heatmap side-by-side\*.
	- [ ] wandb screenshot for 3 randomly chosen validation images (but same images as Q1.6) and heat maps
	- [ ] Report training loss, validation metric1, validation metric2 at the end of training

### Task 2
- [ ] Q2.4 wandb downloaded image of training loss vs iterations
- [ ] Q2.4 wandb downloaded image of test mAP vs iterations plot
- [ ] Q2.4 screenshot for class-wise APs vs iterations for 3 or more classes
- [ ] Q2.4 screenshot of images with predicted boxes for the first logged epoch
- [ ] Q2.4 screenshot of images with predicted boxes for the last logged epoch (~5 epochs)
- [ ] Q2.4 report final classwise APs on the test set and mAP on the test set

## Other Data
- [ ] code folder
