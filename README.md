# EVA-8_Phase-1_Assignment-7
This is the assignment of 7th session in Phase-1 of EVA-8 from TSAI

## Introduction

### Objective
Objective of this assignment is to train resNET18 architecture on [CIFAR10 Dataset](http://yann.lecun.com/exdb/mnist/) and should adhare to following conditions:
1. All major code should in another auxelary repository which will be cloned in this notebook. [link to the repo](https://github.com/devdastl/eva8_source)
2. Need to train the model for 20 epochs.
3. Main colab notebook should not contain any function or class defination. Instead it should import everything from this auxelary repo.
4. Generate loss-accuracy curve for the trained model.
5. Plot misclassified images, also generate gradcam for misclassified images.
6. 
### Getting started
It is very easy to get started with this assignment, just follow below mentioned steps:
1. Open assignment 7 notebook in google colab.
2. Run first cell to clone auxelary repo into the current runtime of colab.
3. Note that deleting runtime can reset and delete cloned repository.
4. 
## Data representation
In this assignment I am using [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) with this dataset I am applying following augmentation on the top:
1. `HorizontalFlip` - Fliping the image along horizontal axis.
2. `ShiftscaleRotate` - Perform transformation randomly, this transformation includes shifting of image, scaling and rotating.
3. `CoarseDropOut` - Overlay a rectangle patch(half the size of original image) on a image randomly. (simulate object hindarence)
4. `ColorJitter` - Randomly changes the brightness, contrast, and saturation of an image. (simulate lighting condition)
5. `ToGray` - Randomly change RBG to gray-scale. 
6. `Normalize` - Normalize image i.e. zero centring (zero mean) and scaling (one std)

Below is the graph representing the input training dataset after appling all augmentations c .
![Alt text](report/data_6.JPG?raw=true "model architecture")

## Model representation
As mentioned, in this assignment we are training restNET18 architecture which has 18 convolution layers. You can find the model architecture defination in [auxlary repo](https://github.com/devdastl/eva8_source/blob/main/models/resnet.py).

Below is a snippet of model architecture summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
```

As shown above, this model contains 4 major layers `32x32x64 -> 16x16x128 -> 8x8x256 -> 4x4x512`

## Training logs
Below is the snippet of last 5 epochs of training log:
```
EPOCH: 15
Loss=0.3357871174812317 Batch_id=390 Accuracy=88.61: 100%|██████████| 391/391 [00:45<00:00,  8.53it/s]

Test set: Average loss: 0.3140, Accuracy: 8946/10000 (89.46%)

EPOCH: 16
Loss=0.5318164825439453 Batch_id=390 Accuracy=88.53: 100%|██████████| 391/391 [00:45<00:00,  8.53it/s]

Test set: Average loss: 0.3132, Accuracy: 8958/10000 (89.58%)

EPOCH: 17
Loss=0.3865199685096741 Batch_id=390 Accuracy=88.97: 100%|██████████| 391/391 [00:45<00:00,  8.55it/s]

Test set: Average loss: 0.3144, Accuracy: 8955/10000 (89.55%)

EPOCH: 18
Loss=0.20548447966575623 Batch_id=390 Accuracy=89.06: 100%|██████████| 391/391 [00:46<00:00,  8.50it/s]

Test set: Average loss: 0.3132, Accuracy: 8957/10000 (89.57%)

EPOCH: 19
Loss=0.36903080344200134 Batch_id=390 Accuracy=89.44: 100%|██████████| 391/391 [00:46<00:00,  8.50it/s]
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19
generating mis-classified images for epoch 19

Test set: Average loss: 0.3115, Accuracy: 8954/10000 (89.54%)
```
## Results
As a result, we will look into generated graphs, mis-classified images and GradCAM of those images.

#### Accuracy-Loss graph
Below is the image showing Accuracy-loss curve for validation dataset.
![Alt text](report/graph.png?raw=true "model architecture")

#### Mis-classified images
Below is the plot of mis-classified images while validating on 20th epoch.
![Alt text](report/misclassified.png?raw=true "model architecture")

#### GradCAM of mis-classified images
Below are GradCAM images for 10 mis-classified images. We have captured GradCAM in two different layers

GradCAM at Layer 3                                   | GradCAM at Layer 4
:---------------------------------------------------:|:--------------------------------------------------:
![Alt text](report/misclass_grad_3.png?raw=true "")  | ![Alt text](report/misclass_grad_4.png?raw=true "")

## Conclusion

