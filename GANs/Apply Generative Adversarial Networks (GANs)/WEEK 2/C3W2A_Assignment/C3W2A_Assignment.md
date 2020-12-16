# U-Net

### Goals
In this notebook, you're going to implement a U-Net for a biomedical imaging segmentation task. Specifically, you're going to be labeling neurons, so one might call this a neural neural network! ;) 

Note that this is not a GAN, generative model, or unsupervised learning task. This is a supervised learning task, so there's only one correct answer (like a classifier!) You will see how this component underlies the Generator component of Pix2Pix in the next notebook this week.

### Learning Objectives
1.   Implement your own U-Net.
2.   Observe your U-Net's performance on a challenging segmentation task.


## Getting Started
You will start by importing libraries, defining a visualization function, and getting the neural dataset that you will be using.

#### Dataset
For this notebook, you will be using a dataset of electron microscopy
images and segmentation data. The information about the dataset you'll be using can be found [here](https://www.ini.uzh.ch/~acardona/data.html)! 

> Arganda-Carreras et al. "Crowdsourcing the creation of image
segmentation algorithms for connectomics". Front. Neuroanat. 2015. https://www.frontiersin.org/articles/10.3389/fnana.2015.00142/full

![dataset example](Neuraldatasetexample.png)



```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    # image_shifted = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```

## U-Net Architecture
Now you can build your U-Net from its components. The figure below is from the paper, [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597), by Ronneberger et al. 2015. It shows the U-Net architecture and how it contracts and then expands.

<!-- "[i]t consists of a contracting path (left side) and an expansive path (right side)" (Renneberger, 2015) -->

![Figure 1 from the paper, U-Net: Convolutional Networks for Biomedical Image Segmentation](https://drive.google.com/uc?export=view&id=1XgJRexE2CmsetRYyTLA7L8dsEwx7aQZY)

In other words, images are first fed through many convolutional layers which reduce height and width while increasing the channels, which the authors refer to as the "contracting path." For example, a set of two 2 x 2 convolutions with a stride of 2, will take a 1 x 28 x 28 (channels, height, width) grayscale image and result in a 2 x 14 x 14 representation. The "expanding path" does the opposite, gradually growing the image with fewer and fewer channels.

## Contracting Path
You will first implement the contracting blocks for the contracting path. This path is the encoder section of the U-Net, which has several downsampling steps as part of it. The authors give more detail of the remaining parts in the following paragraph from the paper (Renneberger, 2015):

>The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3 x 3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2 x 2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels.

<details>
<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">ContractingBlock</font></code></b>
</font>
</summary>

1.    Both convolutions should use 3 x 3 kernels.
2.    The max pool should use a 2 x 2 kernel with a stride 2.
</details>


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: ContractingBlock
class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ContractingBlock, self).__init__()
        # You want to double the number of channels in the first convolution
        # and keep the same number of channels in the second.
        #### START CODE HERE ####
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #### END CODE HERE ####

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x
    
    # Required for grading
    def get_self(self):
        return self
```


```python
#UNIT TEST
def test_contracting_block(test_samples=100, test_channels=10, test_size=50):
    test_block = ContractingBlock(test_channels)
    test_in = torch.randn(test_samples, test_channels, test_size, test_size)
    test_out_conv1 = test_block.conv1(test_in)
    # Make sure that the first convolution has the right shape
    assert tuple(test_out_conv1.shape) == (test_samples, test_channels * 2, test_size - 2, test_size - 2)
    # Make sure that the right activation is used
    assert torch.all(test_block.activation(test_out_conv1) >= 0)
    assert torch.max(test_block.activation(test_out_conv1)) >= 1
    test_out_conv2 = test_block.conv2(test_out_conv1)
    # Make sure that the second convolution has the right shape
    assert tuple(test_out_conv2.shape) == (test_samples, test_channels * 2, test_size - 4, test_size - 4)
    test_out = test_block(test_in)
    # Make sure that the pooling has the right shape
    assert tuple(test_out.shape) == (test_samples, test_channels * 2, test_size // 2 - 2, test_size // 2 - 2)

test_contracting_block()
test_contracting_block(10, 9, 8)
print("Success!")
```

    Success!


## Expanding Path
Next, you will implement the expanding blocks for the expanding path. This is the decoding section of U-Net which has several upsampling steps as part of it. In order to do this, you'll also need to write a crop function. This is so you can crop the image from the *contracting path* and concatenate it to the current image on the expanding path—this is to form a skip connection. Again, the details are from the paper (Renneberger, 2015):

>Every step in the expanding path consists of an upsampling of the feature map followed by a 2 x 2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3 x 3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution.

<!-- so that the expanding block can resize the input from the contracting block can have the same size as the input from the previous layer -->

*Fun fact: later models based on this architecture often use padding in the convolutions to prevent the size of the image from changing outside of the upsampling / downsampling steps!*

<details>
<summary>
<font size="3" color="green">
<b>Optional hint for <code><font size="4">ExpandingBlock</font></code></b>
</font>
</summary>

1.    The concatenation means the number of channels goes back to being input_channels, so you need to halve it again for the next convolution.
</details>


```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: crop
def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    # There are many ways to implement this crop function, but it's what allows
    # the skip connection to function as intended with two differently sized images!
    #### START CODE HERE ####
    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - new_shape[2] // 2
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - new_shape[3] // 2
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]
    #### END CODE HERE ####
    return cropped_image
```


```python
#UNIT TEST
def test_expanding_block_crop(test_samples=100, test_channels=10, test_size=100):
    # Make sure that the crop function is the right shape
    skip_con_x = torch.randn(test_samples, test_channels, test_size + 6, test_size + 6)
    x = torch.randn(test_samples, test_channels, test_size, test_size)
    cropped = crop(skip_con_x, x.shape)
    assert tuple(cropped.shape) == (test_samples, test_channels, test_size, test_size)

    # Make sure that the crop function takes the right area
    test_meshgrid = torch.meshgrid([torch.arange(0, test_size), torch.arange(0, test_size)])
    test_meshgrid = test_meshgrid[0] + test_meshgrid[1]
    test_meshgrid = test_meshgrid[None, None, :, :].float()
    cropped = crop(test_meshgrid, torch.Size([1, 1, test_size // 2, test_size // 2]))
    assert cropped.max() == (test_size - 1) * 2 - test_size // 2
    assert cropped.min() == test_size // 2
    assert cropped.mean() == test_size - 1

    test_meshgrid = torch.meshgrid([torch.arange(0, test_size), torch.arange(0, test_size)])
    test_meshgrid = test_meshgrid[0] + test_meshgrid[1]
    crop_size = 5
    test_meshgrid = test_meshgrid[None, None, :, :].float()
    cropped = crop(test_meshgrid, torch.Size([1, 1, crop_size, crop_size]))
    assert cropped.max() <= (test_size + crop_size - 1) and cropped.max() >= test_size - 1
    assert cropped.min() >= (test_size - crop_size - 1) and cropped.min() <= test_size - 1
    assert abs(cropped.mean() - test_size) <= 2

test_expanding_block_crop()
print("Success!")
```

    Success!



```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: ExpandingBlock
class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # "followed by a 2x2 convolution that halves the number of feature channels"
        # "a concatenation with the correspondingly cropped feature map from the contracting path"
        # "and two 3x3 convolutions"
        #### START CODE HERE ####
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3, stride=1)
        #### END CODE HERE ####
        self.activation = nn.ReLU() # "each followed by a ReLU"
 
    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        return x
    
    # Required for grading
    def get_self(self):
        return self
```


```python
#UNIT TEST
def test_expanding_block(test_samples=100, test_channels=10, test_size=50):
    test_block = ExpandingBlock(test_channels)
    skip_con_x = torch.randn(test_samples, test_channels // 2, test_size * 2 + 6, test_size * 2 + 6)
    x = torch.randn(test_samples, test_channels, test_size, test_size)
    x = test_block.upsample(x)
    x = test_block.conv1(x)
    # Make sure that the first convolution produces the right shape
    assert tuple(x.shape) == (test_samples, test_channels // 2,  test_size * 2 - 1, test_size * 2 - 1)
    orginal_x = crop(skip_con_x, x.shape)
    x = torch.cat([x, orginal_x], axis=1)
    x = test_block.conv2(x)
    # Make sure that the second convolution produces the right shape
    assert tuple(x.shape) == (test_samples, test_channels // 2,  test_size * 2 - 3, test_size * 2 - 3)
    x = test_block.conv3(x)
    # Make sure that the final convolution produces the right shape
    assert tuple(x.shape) == (test_samples, test_channels // 2,  test_size * 2 - 5, test_size * 2 - 5)
    x = test_block.activation(x)

test_expanding_block()
print("Success!")
```

    Success!


## Final Layer
Now you will write the final feature mapping block, which takes in a tensor with arbitrarily many tensors and produces a tensor with the same number of pixels but with the correct number of output channels. From the paper (Renneberger, 2015):

>At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.



```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: FeatureMapBlock
class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a UNet - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        #### START CODE HERE ####
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x
```


```python
# UNIT TEST
assert tuple(FeatureMapBlock(10, 60)(torch.randn(1, 10, 10, 10)).shape) == (1, 60, 10, 10)
print("Success!")
```

    Success!


## U-Net

Now you can put it all together! Here, you'll write a `UNet` class which will combine a series of the three kinds of blocks you've implemented.


```python
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: UNet
class UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(UNet, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        # Keep in mind that the expand function takes two inputs, 
        # both with the same number of channels. 
        #### START CODE HERE ####
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expand1(x4, x3)
        x6 = self.expand2(x5, x2)
        x7 = self.expand3(x6, x1)
        x8 = self.expand4(x7, x0)
        xn = self.downfeature(x8)
        #### END CODE HERE ####
        return xn
```


```python
#UNIT TEST
test_unet = UNet(1, 3)
assert tuple(test_unet(torch.randn(1, 1, 256, 256)).shape) == (1, 3, 117, 117)
print("Success!")
```

    Success!


## Training

Finally, you will put this into action!
Remember that these are your parameters:
  *   criterion: the loss function
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   input_dim: the number of channels of the input image
  *   label_dim: the number of channels of the output image
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   initial_shape: the size of the input image (in pixels)
  *   target_shape: the size of the output image (in pixels)
  *   device: the device type

This should take only a few minutes to train!


```python
import torch.nn.functional as F
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
input_dim = 1
label_dim = 1
display_step = 20
batch_size = 4
lr = 0.0002
initial_shape = 512
target_shape = 373
device = 'cuda'
```


```python
from skimage import io
import numpy as np
volumes = torch.Tensor(io.imread('train-volume.tif'))[:, None, :, :] / 255
labels = torch.Tensor(io.imread('train-labels.tif', plugin="tifffile"))[:, None, :, :] / 255
labels = crop(labels, torch.Size([len(labels), 1, target_shape, target_shape]))
dataset = torch.utils.data.TensorDataset(volumes, labels)
```


```python
def train():
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    unet = UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    cur_step = 0

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the image
            real = real.to(device)
            labels = labels.to(device)

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                show_tensor_images(
                    crop(real, torch.Size([len(real), 1, target_shape, target_shape])), 
                    size=(input_dim, target_shape, target_shape)
                )
                show_tensor_images(labels, size=(label_dim, target_shape, target_shape))
                show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape))
            cur_step += 1

train()
```


    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 0: Step 0: U-Net loss: 0.6775259375572205



![png](output_22_2.png)



![png](output_22_3.png)



![png](output_22_4.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 2: Step 20: U-Net loss: 0.5139520764350891



![png](output_22_10.png)



![png](output_22_11.png)



![png](output_22_12.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 5: Step 40: U-Net loss: 0.4868086278438568



![png](output_22_20.png)



![png](output_22_21.png)



![png](output_22_22.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 7: Step 60: U-Net loss: 0.4124978482723236



![png](output_22_28.png)



![png](output_22_29.png)



![png](output_22_30.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 10: Step 80: U-Net loss: 0.3517676889896393



![png](output_22_38.png)



![png](output_22_39.png)



![png](output_22_40.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 12: Step 100: U-Net loss: 0.3736760914325714



![png](output_22_46.png)



![png](output_22_47.png)



![png](output_22_48.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 15: Step 120: U-Net loss: 0.33920833468437195



![png](output_22_56.png)



![png](output_22_57.png)



![png](output_22_58.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 17: Step 140: U-Net loss: 0.35646721720695496



![png](output_22_64.png)



![png](output_22_65.png)



![png](output_22_66.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 20: Step 160: U-Net loss: 0.3747536540031433



![png](output_22_74.png)



![png](output_22_75.png)



![png](output_22_76.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 22: Step 180: U-Net loss: 0.36335495114326477



![png](output_22_82.png)



![png](output_22_83.png)



![png](output_22_84.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 25: Step 200: U-Net loss: 0.382778525352478



![png](output_22_92.png)



![png](output_22_93.png)



![png](output_22_94.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 27: Step 220: U-Net loss: 0.3369518518447876



![png](output_22_100.png)



![png](output_22_101.png)



![png](output_22_102.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 30: Step 240: U-Net loss: 0.33116501569747925



![png](output_22_110.png)



![png](output_22_111.png)



![png](output_22_112.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 32: Step 260: U-Net loss: 0.33346712589263916



![png](output_22_118.png)



![png](output_22_119.png)



![png](output_22_120.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 35: Step 280: U-Net loss: 0.3975258767604828



![png](output_22_128.png)



![png](output_22_129.png)



![png](output_22_130.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 37: Step 300: U-Net loss: 0.2966267168521881



![png](output_22_136.png)



![png](output_22_137.png)



![png](output_22_138.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 40: Step 320: U-Net loss: 0.28023529052734375



![png](output_22_146.png)



![png](output_22_147.png)



![png](output_22_148.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 42: Step 340: U-Net loss: 0.26964521408081055



![png](output_22_154.png)



![png](output_22_155.png)



![png](output_22_156.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 45: Step 360: U-Net loss: 0.3560916483402252



![png](output_22_164.png)



![png](output_22_165.png)



![png](output_22_166.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 47: Step 380: U-Net loss: 0.315801203250885



![png](output_22_172.png)



![png](output_22_173.png)



![png](output_22_174.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 50: Step 400: U-Net loss: 0.2753041982650757



![png](output_22_182.png)



![png](output_22_183.png)



![png](output_22_184.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 52: Step 420: U-Net loss: 0.2646105885505676



![png](output_22_190.png)



![png](output_22_191.png)



![png](output_22_192.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 55: Step 440: U-Net loss: 0.23472483456134796



![png](output_22_200.png)



![png](output_22_201.png)



![png](output_22_202.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 57: Step 460: U-Net loss: 0.20354710519313812



![png](output_22_208.png)



![png](output_22_209.png)



![png](output_22_210.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 60: Step 480: U-Net loss: 0.32372602820396423



![png](output_22_218.png)



![png](output_22_219.png)



![png](output_22_220.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 62: Step 500: U-Net loss: 0.2160012573003769



![png](output_22_226.png)



![png](output_22_227.png)



![png](output_22_228.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 65: Step 520: U-Net loss: 0.2170095294713974



![png](output_22_236.png)



![png](output_22_237.png)



![png](output_22_238.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 67: Step 540: U-Net loss: 0.20949456095695496



![png](output_22_244.png)



![png](output_22_245.png)



![png](output_22_246.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 70: Step 560: U-Net loss: 0.22233670949935913



![png](output_22_254.png)



![png](output_22_255.png)



![png](output_22_256.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 72: Step 580: U-Net loss: 0.21761304140090942



![png](output_22_262.png)



![png](output_22_263.png)



![png](output_22_264.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 75: Step 600: U-Net loss: 0.2121521383523941



![png](output_22_272.png)



![png](output_22_273.png)



![png](output_22_274.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 77: Step 620: U-Net loss: 0.20392905175685883



![png](output_22_280.png)



![png](output_22_281.png)



![png](output_22_282.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 80: Step 640: U-Net loss: 0.20235908031463623



![png](output_22_290.png)



![png](output_22_291.png)



![png](output_22_292.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 82: Step 660: U-Net loss: 0.216666579246521



![png](output_22_298.png)



![png](output_22_299.png)



![png](output_22_300.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 85: Step 680: U-Net loss: 0.16816169023513794



![png](output_22_308.png)



![png](output_22_309.png)



![png](output_22_310.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 87: Step 700: U-Net loss: 0.1759365200996399



![png](output_22_316.png)



![png](output_22_317.png)



![png](output_22_318.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 90: Step 720: U-Net loss: 0.16324247419834137



![png](output_22_326.png)



![png](output_22_327.png)



![png](output_22_328.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 92: Step 740: U-Net loss: 0.13896183669567108



![png](output_22_334.png)



![png](output_22_335.png)



![png](output_22_336.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 95: Step 760: U-Net loss: 0.144816592335701



![png](output_22_344.png)



![png](output_22_345.png)



![png](output_22_346.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 97: Step 780: U-Net loss: 0.1488358974456787



![png](output_22_352.png)



![png](output_22_353.png)



![png](output_22_354.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 100: Step 800: U-Net loss: 0.1367058902978897



![png](output_22_362.png)



![png](output_22_363.png)



![png](output_22_364.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 102: Step 820: U-Net loss: 0.13587553799152374



![png](output_22_370.png)



![png](output_22_371.png)



![png](output_22_372.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 105: Step 840: U-Net loss: 0.12533947825431824



![png](output_22_380.png)



![png](output_22_381.png)



![png](output_22_382.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 107: Step 860: U-Net loss: 0.13028991222381592



![png](output_22_388.png)



![png](output_22_389.png)



![png](output_22_390.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 110: Step 880: U-Net loss: 0.11069062352180481



![png](output_22_398.png)



![png](output_22_399.png)



![png](output_22_400.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 112: Step 900: U-Net loss: 0.12030952423810959



![png](output_22_406.png)



![png](output_22_407.png)



![png](output_22_408.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 115: Step 920: U-Net loss: 0.11645384877920151



![png](output_22_416.png)



![png](output_22_417.png)



![png](output_22_418.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 117: Step 940: U-Net loss: 0.10150367021560669



![png](output_22_424.png)



![png](output_22_425.png)



![png](output_22_426.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 120: Step 960: U-Net loss: 0.09465138614177704



![png](output_22_434.png)



![png](output_22_435.png)



![png](output_22_436.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 122: Step 980: U-Net loss: 0.10968445241451263



![png](output_22_442.png)



![png](output_22_443.png)



![png](output_22_444.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 125: Step 1000: U-Net loss: 0.10357336699962616



![png](output_22_452.png)



![png](output_22_453.png)



![png](output_22_454.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 127: Step 1020: U-Net loss: 0.08627662062644958



![png](output_22_460.png)



![png](output_22_461.png)



![png](output_22_462.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 130: Step 1040: U-Net loss: 0.09717156738042831



![png](output_22_470.png)



![png](output_22_471.png)



![png](output_22_472.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 132: Step 1060: U-Net loss: 0.07982124388217926



![png](output_22_478.png)



![png](output_22_479.png)



![png](output_22_480.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 135: Step 1080: U-Net loss: 0.08297519385814667



![png](output_22_488.png)



![png](output_22_489.png)



![png](output_22_490.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 137: Step 1100: U-Net loss: 0.08517942577600479



![png](output_22_496.png)



![png](output_22_497.png)



![png](output_22_498.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 140: Step 1120: U-Net loss: 0.08507516235113144



![png](output_22_506.png)



![png](output_22_507.png)



![png](output_22_508.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 142: Step 1140: U-Net loss: 0.06828087568283081



![png](output_22_514.png)



![png](output_22_515.png)



![png](output_22_516.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 145: Step 1160: U-Net loss: 0.0753111019730568



![png](output_22_524.png)



![png](output_22_525.png)



![png](output_22_526.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 147: Step 1180: U-Net loss: 0.06500156968832016



![png](output_22_532.png)



![png](output_22_533.png)



![png](output_22_534.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 150: Step 1200: U-Net loss: 0.06935954093933105



![png](output_22_542.png)



![png](output_22_543.png)



![png](output_22_544.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 152: Step 1220: U-Net loss: 0.06547334790229797



![png](output_22_550.png)



![png](output_22_551.png)



![png](output_22_552.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 155: Step 1240: U-Net loss: 0.06446420401334763



![png](output_22_560.png)



![png](output_22_561.png)



![png](output_22_562.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 157: Step 1260: U-Net loss: 0.06240769475698471



![png](output_22_568.png)



![png](output_22_569.png)



![png](output_22_570.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 160: Step 1280: U-Net loss: 0.06167719140648842



![png](output_22_578.png)



![png](output_22_579.png)



![png](output_22_580.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 162: Step 1300: U-Net loss: 0.06406264752149582



![png](output_22_586.png)



![png](output_22_587.png)



![png](output_22_588.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 165: Step 1320: U-Net loss: 0.05241549760103226



![png](output_22_596.png)



![png](output_22_597.png)



![png](output_22_598.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 167: Step 1340: U-Net loss: 0.06495900452136993



![png](output_22_604.png)



![png](output_22_605.png)



![png](output_22_606.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 170: Step 1360: U-Net loss: 0.06130800023674965



![png](output_22_614.png)



![png](output_22_615.png)



![png](output_22_616.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 172: Step 1380: U-Net loss: 0.052497316151857376



![png](output_22_622.png)



![png](output_22_623.png)



![png](output_22_624.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 175: Step 1400: U-Net loss: 0.04908054694533348



![png](output_22_632.png)



![png](output_22_633.png)



![png](output_22_634.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 177: Step 1420: U-Net loss: 0.04963886737823486



![png](output_22_640.png)



![png](output_22_641.png)



![png](output_22_642.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 180: Step 1440: U-Net loss: 0.04769667237997055



![png](output_22_650.png)



![png](output_22_651.png)



![png](output_22_652.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 182: Step 1460: U-Net loss: 0.05101940035820007



![png](output_22_658.png)



![png](output_22_659.png)



![png](output_22_660.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 185: Step 1480: U-Net loss: 0.04857638105750084



![png](output_22_668.png)



![png](output_22_669.png)



![png](output_22_670.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 187: Step 1500: U-Net loss: 0.0455658994615078



![png](output_22_676.png)



![png](output_22_677.png)



![png](output_22_678.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 190: Step 1520: U-Net loss: 0.05010155588388443



![png](output_22_686.png)



![png](output_22_687.png)



![png](output_22_688.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 192: Step 1540: U-Net loss: 0.04758180305361748



![png](output_22_694.png)



![png](output_22_695.png)



![png](output_22_696.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 195: Step 1560: U-Net loss: 0.04857000708580017



![png](output_22_704.png)



![png](output_22_705.png)



![png](output_22_706.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    Epoch 197: Step 1580: U-Net loss: 0.044229473918676376



![png](output_22_712.png)



![png](output_22_713.png)



![png](output_22_714.png)


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=8.0), HTML(value='')))


    



```python

```
