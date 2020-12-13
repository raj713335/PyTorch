# Your First GAN

### Goal
In this notebook, you're going to create your first generative adversarial network (GAN) for this course! Specifically, you will build and train a GAN that can generate hand-written images of digits (0-9). You will be using PyTorch in this specialization, so if you're not familiar with this framework, you may find the [PyTorch documentation](https://pytorch.org/docs/stable/index.html) useful. The hints will also often include links to relevant documentation.

### Learning Objectives
1.   Build the generator and discriminator components of a GAN from scratch.
2.   Create generator and discriminator loss functions.
3.   Train your GAN and visualize the generated images.


## Getting Started
You will begin by importing some useful packages and the dataset you will use to build and train your GAN. You are also provided with a visualizer function to help you investigate the images your GAN will create.



```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```

#### MNIST Dataset
The training images your discriminator will be using is from a dataset called [MNIST](http://yann.lecun.com/exdb/mnist/). It contains 60,000 images of handwritten digits, from 0 to 9, like these:

![MNIST Digits](MnistExamples.png)

You may notice that the images are quite pixelated -- this is because they are all only 28 x 28! The small size of its images makes MNIST ideal for simple training. Additionally, these images are also in black-and-white so only one dimension, or "color channel", is needed to represent them (more on this later in the course).

#### Tensor
You will represent the data using [tensors](https://pytorch.org/docs/stable/tensors.html). Tensors are a generalization of matrices: for example, a stack of three matrices with the amounts of red, green, and blue at different locations in a 64 x 64 pixel image is a tensor with the shape 3 x 64 x 64.

Tensors are easy to manipulate and supported by [PyTorch](https://pytorch.org/), the machine learning library you will be using. Feel free to explore them more, but you can imagine these as multi-dimensional matrices or vectors!

#### Batches
While you could train your model after generating one image, it is extremely inefficient and leads to less stable training. In GANs, and in machine learning in general, you will process multiple images per training step. These are called batches.

This means that your generator will generate an entire batch of images and receive the discriminator's feedback on each before updating the model. The same goes for the discriminator, it will calculate its loss on the entire batch of generated images as well as on the reals before the model is updated.

## Generator
The first step is to build the generator component.

You will start by creating a function to make a single layer/block for the generator's neural network. Each block should include a [linear transformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) to map to another shape, a [batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) for stabilization, and finally a non-linear activation function (you use a [ReLU here](https://pytorch.org/docs/master/generated/torch.nn.ReLU.html)) so the output can be transformed in complex ways. You will learn more about activations and batch normalization later in the course.


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_generator_block
def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    #print(input_dim,output_dim)
    return nn.Sequential(
        # Hint: Replace all of the "None" with the appropriate dimensions.
        # The documentation may be useful if you're less familiar with PyTorch:
        # https://pytorch.org/docs/stable/nn.html.
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
        #### END CODE HERE ####
    )
```


```python
# Verify the generator block function
def test_gen_block(in_features, out_features, num_test=1000):
    block = get_generator_block(in_features, out_features)

    # Check the three parts
    assert len(block) == 3
    assert type(block[0]) == nn.Linear
    assert type(block[1]) == nn.BatchNorm1d
    assert type(block[2]) == nn.ReLU
    
    # Check the output shape
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)
    assert test_output.std() > 0.55
    assert test_output.std() < 0.65

test_gen_block(25, 12)
test_gen_block(15, 28)
print("Success!")
```

    Success!


Now you can build the generator class. It will take 3 values:

*   The noise vector dimension
*   The image dimension
*   The initial hidden dimension

Using these values, the generator will build a neural network with 5 layers/blocks. Beginning with the noise vector, the generator will apply non-linear transformations via the block function until the tensor is mapped to the size of the image to be outputted (the same size as the real images from MNIST). You will need to fill in the code for final layer since it is different than the others. The final layer does not need a normalization or activation function, but does need to be scaled with a [sigmoid function](https://pytorch.org/docs/master/generated/torch.nn.Sigmoid.html). 

Finally, you are given a forward pass function that takes in a noise vector and generates an image of the output dimension using your neural network.

<details>

<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">Generator</font></code></b>
</font>
</summary>

1. The output size of the final linear transformation should be im_dim, but remember you need to scale the outputs between 0 and 1 using the sigmoid function.
2. [nn.Linear](https://pytorch.org/docs/master/generated/torch.nn.Linear.html) and [nn.Sigmoid](https://pytorch.org/docs/master/generated/torch.nn.Sigmoid.html) will be useful here. 
</details>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Generator
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            # There is a dropdown with hints if you need them! 
            #### START CODE HERE ####
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()

            #### END CODE HERE ####
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen
```


```python
# Verify the generator class
def test_generator(z_dim, im_dim, hidden_dim, num_test=10000):
    gen = Generator(z_dim, im_dim, hidden_dim).get_gen()
    
    # Check there are six modules in the sequential part
    assert len(gen) == 6
    test_input = torch.randn(num_test, z_dim)
    test_output = gen(test_input)

    # Check that the output shape is correct
    assert tuple(test_output.shape) == (num_test, im_dim)
    assert test_output.max() < 1, "Make sure to use a sigmoid"
    assert test_output.min() > 0, "Make sure to use a sigmoid"
    assert test_output.std() > 0.05, "Don't use batchnorm here"
    assert test_output.std() < 0.15, "Don't use batchnorm here"

test_generator(5, 10, 20)
test_generator(20, 8, 24)
print("Success!")
```

    Success!


## Noise
To be able to use your generator, you will need to be able to create noise vectors. The noise vector z has the important role of making sure the images generated from the same class don't all look the same -- think of it as a random seed. You will generate it randomly using PyTorch by sampling random numbers from the normal distribution. Since multiple images will be processed per pass, you will generate all the noise vectors at once.

Note that whenever you create a new tensor using torch.ones, torch.zeros, or torch.randn, you either need to create it on the target device, e.g. `torch.ones(3, 3, device=device)`, or move it onto the target device using `torch.ones(3, 3).to(device)`. You do not need to do this if you're creating a tensor by manipulating another tensor or by using a variation that defaults the device to the input, such as `torch.ones_like`. In general, use `torch.ones_like` and `torch.zeros_like` instead of `torch.ones` or `torch.zeros` where possible.

<details>

<summary>
<font size="3" color="green">
<b>Optional hint for <code><font size="4">get_noise</font></code></b>
</font>
</summary>

1. 
You will probably find [torch.randn](https://pytorch.org/docs/master/generated/torch.randn.html) useful here.
</details>


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_noise
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device 
    # argument to the function you use to generate the noise.
    #### START CODE HERE ####
    return torch.randn(n_samples, z_dim, device=device)
    #### END CODE HERE ####
```


```python
# Verify the noise vector function
def test_get_noise(n_samples, z_dim, device='cpu'):
    noise = get_noise(n_samples, z_dim, device)
    
    # Make sure a normal distribution was used
    assert tuple(noise.shape) == (n_samples, z_dim)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)

test_get_noise(1000, 100, 'cpu')
if torch.cuda.is_available():
    test_get_noise(1000, 32, 'cuda')
print("Success!")
```

    Success!


## Discriminator
The second component that you need to construct is the discriminator. As with the generator component, you will start by creating a function that builds a neural network block for the discriminator.

*Note: You use leaky ReLUs to prevent the "dying ReLU" problem, which refers to the phenomenon where the parameters stop changing due to consistently negative values passed to a ReLU, which result in a zero gradient. You will learn more about this in the following lectures!* 


REctified Linear Unit (ReLU) |  Leaky ReLU
:-------------------------:|:-------------------------:
![](relu-graph.png)  |  ![](lrelu-graph.png)






```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_discriminator_block
def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
        
        #### END CODE HERE ####
    )
```


```python
# Verify the discriminator block function
def test_disc_block(in_features, out_features, num_test=10000):
    block = get_discriminator_block(in_features, out_features)

    # Check there are two parts
    assert len(block) == 2
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)

    # Check that the shape is right
    assert tuple(test_output.shape) == (num_test, out_features)
    
    # Check that the LeakyReLU slope is about 0.2
    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5

test_disc_block(25, 12)
test_disc_block(15, 28)
print("Success!")
```

    Success!


Now you can use these blocks to make a discriminator! The discriminator class holds 2 values:

*   The image dimension
*   The hidden dimension

The discriminator will build a neural network with 4 layers. It will start with the image tensor and transform it until it returns a single number (1-dimension tensor) output. This output classifies whether an image is fake or real. Note that you do not need a sigmoid after the output layer since it is included in the loss function. Finally, to use your discrimator's neural network you are given a forward pass function that takes in an image tensor to be classified.



```python
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            # Hint: You want to transform the final output into a single value,
            #       so add one more linear map.
            #### START CODE HERE ####
            nn.Linear(hidden_dim, 1)

            #### END CODE HERE ####
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    
    # Needed for grading
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc
```


```python
# Verify the discriminator class
def test_discriminator(z_dim, hidden_dim, num_test=100):
    
    disc = Discriminator(z_dim, hidden_dim).get_disc()

    # Check there are three parts
    assert len(disc) == 4

    # Check the linear layer is correct
    test_input = torch.randn(num_test, z_dim)
    test_output = disc(test_input)
    assert tuple(test_output.shape) == (num_test, 1)
    
    # Make sure there's no sigmoid
    assert test_input.max() > 1
    assert test_input.min() < -1

test_discriminator(5, 10)
test_discriminator(20, 8)
print("Success!")
```

    Success!


## Training
Now you can put it all together!
First, you will set your parameters:
  *   criterion: the loss function
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   device: the device type, here using a GPU (which runs CUDA), not CPU

Next, you will load the MNIST dataset as tensors using a dataloader.




```python
# Set your parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

### DO NOT EDIT ###
device = 'cuda'
```

Now, you can initialize your generator, discriminator, and optimizers. Note that each optimizer only takes the parameters of one particular model, since we want each optimizer to optimize only one of the models.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
```

Before you train your GAN, you will need to create functions to calculate the discriminator's loss and the generator's loss. This is how the discriminator and generator will know how they are doing and improve themselves. Since the generator is needed when calculating the discriminator's loss, you will need to call .detach() on the generator result to ensure that only the discriminator is updated!

Remember that you have already defined a loss function earlier (`criterion`) and you are encouraged to use `torch.ones_like` and `torch.zeros_like` instead of `torch.ones` or `torch.zeros`. If you use `torch.ones` or `torch.zeros`, you'll need to pass `device=device` to them.


```python
# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_disc_loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images. 
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image 
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a 
    #            'ground truth' tensor in order to calculate the loss. 
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     Note: Please do not use concatenation in your solution. The tests are being updated to 
    #           support this, but for now, average the two losses as described in step (4).
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    #### START CODE HERE ####
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2

    #### END CODE HERE ####
    return disc_loss
```


```python
def test_disc_reasonable(num_images=10):
    # Don't use explicit casts to cuda - use the device argument
    import inspect, re
    lines = inspect.getsource(get_disc_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None
    
    z_dim = 64
    gen = torch.zeros_like
    disc = lambda x: x.mean(1)[:, None]
    criterion = torch.mul # Multiply
    real = torch.ones(num_images, z_dim)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(disc_loss.mean() - 0.5) < 1e-5)
    
    gen = torch.ones_like
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, z_dim)
    assert torch.all(torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')) < 1e-5)
    
    gen = lambda x: torch.ones(num_images, 10)
    disc = lambda x: x.mean(1)[:, None] + 10
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, 10)
    assert torch.all(torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean() - 5) < 1e-5)

    gen = torch.ones_like
    disc = nn.Linear(64, 1, bias=False)
    real = torch.ones(num_images, 64) * 0.5
    disc.weight.data = torch.ones_like(disc.weight.data) * 0.5
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = lambda x, y: torch.sum(x) + torch.sum(y)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean()
    disc_loss.backward()
    assert torch.isclose(torch.abs(disc.weight.grad.mean() - 11.25), torch.tensor(3.75))
    
def test_disc_loss(max_tests = 10):
    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    num_steps = 0
    for real, _ in dataloader:
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradient before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        assert (disc_loss - 0.68).abs() < 0.05

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Check that they detached correctly
        assert gen.gen[0][0].weight.grad is None

        # Update optimizer
        old_weight = disc.disc[0][0].weight.data.clone()
        disc_opt.step()
        new_weight = disc.disc[0][0].weight.data
        
        # Check that some discriminator weights changed
        assert not torch.all(torch.eq(old_weight, new_weight))
        num_steps += 1
        if num_steps >= max_tests:
            break

test_disc_reasonable()
test_disc_loss()
print("Success!")
```

    Success!



```python
# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch of fake images. 
    #           Remember to pass the device argument to the get_noise function.
    #       2) Get the discriminator's prediction of the fake image.
    #       3) Calculate the generator's loss. Remember the generator wants
    #          the discriminator to think that its fake images are real
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!

    #### START CODE HERE ####
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    #### END CODE HERE ####
    return gen_loss
```


```python
def test_gen_reasonable(num_images=10):
    # Don't use explicit casts to cuda - use the device argument
    import inspect, re
    lines = inspect.getsource(get_gen_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None
    
    z_dim = 64
    gen = torch.zeros_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(gen_loss_tensor) < 1e-5)
    #Verify shape. Related to gen_noise parametrization
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)

    gen = torch.ones_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, 1)
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(gen_loss_tensor - 1) < 1e-5)
    #Verify shape. Related to gen_noise parametrization
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)
    

def test_gen_loss(num_images):
    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    
    gen_loss = get_gen_loss(gen, disc, criterion, num_images, z_dim, device)
    
    # Check that the loss is reasonable
    assert (gen_loss - 0.7).abs() < 0.1
    gen_loss.backward()
    old_weight = gen.gen[0][0].weight.clone()
    gen_opt.step()
    new_weight = gen.gen[0][0].weight
    assert not torch.all(torch.eq(old_weight, new_weight))


test_gen_reasonable(10)
test_gen_loss(18)
print("Success!")
```

    Success!


Finally, you can put everything together! For each epoch, you will process the entire dataset in batches. For every batch, you will need to update the discriminator and generator using their loss. Batches are sets of images that will be predicted on before the loss functions are calculated (instead of calculating the loss function after each image). Note that you may see a loss to be greater than 1, this is okay since binary cross entropy loss can be any positive number for a sufficiently confident wrong guess. 

It’s also often the case that the discriminator will outperform the generator, especially at the start, because its job is easier. It's important that neither one gets too good (that is, near-perfect accuracy), which would cause the entire model to stop learning. Balancing the two models is actually remarkably hard to do in a standard GAN and something you will see more of in later lectures and assignments.

After you've submitted a working version with the original architecture, feel free to play around with the architecture if you want to see how different architectural choices can lead to better or worse GANs. For example, consider changing the size of the hidden dimension, or making the networks shallower or deeper by changing the number of layers.

<!-- In addition, be warned that this runs very slowly on a CPU. One way to run this more quickly is to use Google Colab: 

1.   Download the .ipynb
2.   Upload it to Google Drive and open it with Google Colab
3.   Make the runtime type GPU (under “Runtime” -> “Change runtime type” -> Select “GPU” from the dropdown)
4.   Replace `device = "cpu"` with `device = "cuda"`
5.   Make sure your `get_noise` function uses the right device -->

But remember, don’t expect anything spectacular: this is only the first lesson. The results will get better with later lessons as you learn methods to help keep your generator and discriminator at similar levels.

You should roughly expect to see this progression. On a GPU, this should take about 15 seconds per 500 steps, on average, while on CPU it will take roughly 1.5 minutes:
![MNIST Digits](MNIST_Progression.png)


```python
# UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: 

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True # Whether the generator should be tested
gen_loss = False
error = False
for epoch in range(n_epochs):
  
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        ### Update generator ###
        #     Hint: This code will look a lot like the discriminator updates!
        #     These are the steps you will need to complete:
        #       1) Zero out the gradients.
        #       2) Calculate the generator loss, assigning it to gen_loss.
        #       3) Backprop through the generator: update the gradients and optimizer.
        #### START CODE HERE ####
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()
        #### END CODE HERE ####

        # For testing purposes, to check that your code changes the generator weights
        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

```


    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 1, step 500: Generator loss: 1.3957562952041633, discriminator loss: 0.41824104803800605



![png](output_31_4.png)



![png](output_31_5.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 2, step 1000: Generator loss: 1.7552806265354173, discriminator loss: 0.2786566876769066



![png](output_31_9.png)



![png](output_31_10.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 3, step 1500: Generator loss: 2.056151607751848, discriminator loss: 0.15896223971247678



![png](output_31_14.png)



![png](output_31_15.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 4, step 2000: Generator loss: 1.6764541094303145, discriminator loss: 0.22502257880568502



![png](output_31_19.png)



![png](output_31_20.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 5, step 2500: Generator loss: 1.6625128796100608, discriminator loss: 0.20657719141244874



![png](output_31_24.png)



![png](output_31_25.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 6, step 3000: Generator loss: 1.8608701801300052, discriminator loss: 0.1822700653374197



![png](output_31_29.png)



![png](output_31_30.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 7, step 3500: Generator loss: 2.302583043813706, discriminator loss: 0.1400712618380785



![png](output_31_34.png)



![png](output_31_35.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 8, step 4000: Generator loss: 2.7221983804702736, discriminator loss: 0.11245058816671367



![png](output_31_39.png)



![png](output_31_40.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 9, step 4500: Generator loss: 3.136809785842895, discriminator loss: 0.08697137556225064



![png](output_31_44.png)



![png](output_31_45.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 10, step 5000: Generator loss: 3.504528656959534, discriminator loss: 0.06510535230487588



![png](output_31_49.png)



![png](output_31_50.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 11, step 5500: Generator loss: 3.805730495929717, discriminator loss: 0.05233940044417977



![png](output_31_54.png)



![png](output_31_55.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 12, step 6000: Generator loss: 4.007290257930755, discriminator loss: 0.049507093470543655



![png](output_31_59.png)



![png](output_31_60.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 13, step 6500: Generator loss: 4.161701416969298, discriminator loss: 0.050393566932529194



![png](output_31_64.png)



![png](output_31_65.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 14, step 7000: Generator loss: 4.286148863792421, discriminator loss: 0.04843816028907895



![png](output_31_69.png)



![png](output_31_70.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 15, step 7500: Generator loss: 4.523370460510255, discriminator loss: 0.04648638882115483



![png](output_31_74.png)



![png](output_31_75.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 17, step 8000: Generator loss: 4.227649399280545, discriminator loss: 0.05167542887851595



![png](output_31_81.png)



![png](output_31_82.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 18, step 8500: Generator loss: 4.622684229850769, discriminator loss: 0.0483427852503955



![png](output_31_86.png)



![png](output_31_87.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 19, step 9000: Generator loss: 4.496660138130186, discriminator loss: 0.04552671618387105



![png](output_31_91.png)



![png](output_31_92.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 20, step 9500: Generator loss: 4.494158327102658, discriminator loss: 0.051650805000215766



![png](output_31_96.png)



![png](output_31_97.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 21, step 10000: Generator loss: 4.534346714019776, discriminator loss: 0.049008878163993394



![png](output_31_101.png)



![png](output_31_102.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 22, step 10500: Generator loss: 4.467069611549376, discriminator loss: 0.049734368123114124



![png](output_31_106.png)



![png](output_31_107.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 23, step 11000: Generator loss: 4.369014120101929, discriminator loss: 0.05860964015871284



![png](output_31_111.png)



![png](output_31_112.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 24, step 11500: Generator loss: 4.518626810073849, discriminator loss: 0.06603773651272069



![png](output_31_116.png)



![png](output_31_117.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 25, step 12000: Generator loss: 4.1491234536170944, discriminator loss: 0.07600318007171153



![png](output_31_121.png)



![png](output_31_122.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 26, step 12500: Generator loss: 4.122006687164311, discriminator loss: 0.079478247307241



![png](output_31_126.png)



![png](output_31_127.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 27, step 13000: Generator loss: 4.009247925758362, discriminator loss: 0.09308315671235311



![png](output_31_131.png)



![png](output_31_132.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 28, step 13500: Generator loss: 3.8383395452499416, discriminator loss: 0.09523410909622909



![png](output_31_136.png)



![png](output_31_137.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 29, step 14000: Generator loss: 3.815989500522614, discriminator loss: 0.0911934330239893



![png](output_31_141.png)



![png](output_31_142.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 30, step 14500: Generator loss: 3.8707463016510015, discriminator loss: 0.10070136234909287



![png](output_31_146.png)



![png](output_31_147.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 31, step 15000: Generator loss: 3.89845133876801, discriminator loss: 0.10734250236302607



![png](output_31_151.png)



![png](output_31_152.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 33, step 15500: Generator loss: 3.5354170870780917, discriminator loss: 0.13008888967335228



![png](output_31_158.png)



![png](output_31_159.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 34, step 16000: Generator loss: 3.730554688930512, discriminator loss: 0.11978914602100857



![png](output_31_163.png)



![png](output_31_164.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 35, step 16500: Generator loss: 3.750533558845516, discriminator loss: 0.11790555789321676



![png](output_31_168.png)



![png](output_31_169.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 36, step 17000: Generator loss: 3.4589380784034742, discriminator loss: 0.1418039887100459



![png](output_31_173.png)



![png](output_31_174.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 37, step 17500: Generator loss: 3.370402026176452, discriminator loss: 0.13079801913350814



![png](output_31_178.png)



![png](output_31_179.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 38, step 18000: Generator loss: 3.468122571945195, discriminator loss: 0.13503547176718717



![png](output_31_183.png)



![png](output_31_184.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 39, step 18500: Generator loss: 3.2629546742439284, discriminator loss: 0.16066625516116623



![png](output_31_188.png)



![png](output_31_189.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 40, step 19000: Generator loss: 3.3086451630592304, discriminator loss: 0.15042537683248516



![png](output_31_193.png)



![png](output_31_194.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 41, step 19500: Generator loss: 3.25120729494095, discriminator loss: 0.17290936566889287



![png](output_31_198.png)



![png](output_31_199.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 42, step 20000: Generator loss: 3.0807797546386704, discriminator loss: 0.17825241464376446



![png](output_31_203.png)



![png](output_31_204.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 43, step 20500: Generator loss: 3.0357120389938337, discriminator loss: 0.17221854396164415



![png](output_31_208.png)



![png](output_31_209.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 44, step 21000: Generator loss: 3.1098047990798974, discriminator loss: 0.1577930937558414



![png](output_31_213.png)



![png](output_31_214.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 45, step 21500: Generator loss: 3.0906408953666693, discriminator loss: 0.17937957951426528



![png](output_31_218.png)



![png](output_31_219.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 46, step 22000: Generator loss: 3.0440940508842487, discriminator loss: 0.17446485844254514



![png](output_31_223.png)



![png](output_31_224.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 47, step 22500: Generator loss: 3.006882413864135, discriminator loss: 0.19376483684778206



![png](output_31_228.png)



![png](output_31_229.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 49, step 23000: Generator loss: 3.052919278144839, discriminator loss: 0.1837128566205503



![png](output_31_235.png)



![png](output_31_236.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 50, step 23500: Generator loss: 2.8912532486915623, discriminator loss: 0.18220273298025125



![png](output_31_240.png)



![png](output_31_241.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 51, step 24000: Generator loss: 2.952877184391021, discriminator loss: 0.18151734682917584



![png](output_31_245.png)



![png](output_31_246.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 52, step 24500: Generator loss: 2.75950974178314, discriminator loss: 0.20602816034853455



![png](output_31_250.png)



![png](output_31_251.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 53, step 25000: Generator loss: 2.726589030265809, discriminator loss: 0.20617326717078685



![png](output_31_255.png)



![png](output_31_256.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 54, step 25500: Generator loss: 2.8177708635330183, discriminator loss: 0.1777015901356937



![png](output_31_260.png)



![png](output_31_261.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 55, step 26000: Generator loss: 2.8146701049804688, discriminator loss: 0.20350523656606687



![png](output_31_265.png)



![png](output_31_266.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 56, step 26500: Generator loss: 2.7447915043830866, discriminator loss: 0.2012428192347286



![png](output_31_270.png)



![png](output_31_271.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 57, step 27000: Generator loss: 2.9126328268051123, discriminator loss: 0.18733542206883436



![png](output_31_275.png)



![png](output_31_276.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 58, step 27500: Generator loss: 2.6884643020629895, discriminator loss: 0.2111651129126548



![png](output_31_280.png)



![png](output_31_281.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 59, step 28000: Generator loss: 2.725103603363037, discriminator loss: 0.20509056004881848



![png](output_31_285.png)



![png](output_31_286.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 60, step 28500: Generator loss: 2.6442266511917114, discriminator loss: 0.2091303559839725



![png](output_31_290.png)



![png](output_31_291.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 61, step 29000: Generator loss: 2.7977133555412292, discriminator loss: 0.18219305160641674



![png](output_31_295.png)



![png](output_31_296.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 62, step 29500: Generator loss: 2.5894435687065114, discriminator loss: 0.22335313332080844



![png](output_31_300.png)



![png](output_31_301.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 63, step 30000: Generator loss: 2.5438011398315448, discriminator loss: 0.21697981882095307



![png](output_31_305.png)



![png](output_31_306.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 65, step 30500: Generator loss: 2.5261828541755666, discriminator loss: 0.24181503057479853



![png](output_31_312.png)



![png](output_31_313.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 66, step 31000: Generator loss: 2.5936116614341733, discriminator loss: 0.20896324613690365



![png](output_31_317.png)



![png](output_31_318.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 67, step 31500: Generator loss: 2.565617925643921, discriminator loss: 0.236862419694662



![png](output_31_322.png)



![png](output_31_323.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 68, step 32000: Generator loss: 2.6346095561981184, discriminator loss: 0.21515017864108077



![png](output_31_327.png)



![png](output_31_328.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 69, step 32500: Generator loss: 2.7664735608100925, discriminator loss: 0.19562571105360974



![png](output_31_332.png)



![png](output_31_333.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 70, step 33000: Generator loss: 2.6279735083579983, discriminator loss: 0.2252546804845331



![png](output_31_337.png)



![png](output_31_338.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 71, step 33500: Generator loss: 2.6194112524986246, discriminator loss: 0.2190304816365242



![png](output_31_342.png)



![png](output_31_343.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 72, step 34000: Generator loss: 2.6192685637474056, discriminator loss: 0.21840808048844337



![png](output_31_347.png)



![png](output_31_348.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 73, step 34500: Generator loss: 2.510606921195984, discriminator loss: 0.2398007779717447



![png](output_31_352.png)



![png](output_31_353.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 74, step 35000: Generator loss: 2.4928616199493407, discriminator loss: 0.23367412233352666



![png](output_31_357.png)



![png](output_31_358.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 75, step 35500: Generator loss: 2.3953575811386107, discriminator loss: 0.24808615776896462



![png](output_31_362.png)



![png](output_31_363.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 76, step 36000: Generator loss: 2.450585974216463, discriminator loss: 0.24187756162881877



![png](output_31_367.png)



![png](output_31_368.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 77, step 36500: Generator loss: 2.361032798767091, discriminator loss: 0.2687371503710748



![png](output_31_372.png)



![png](output_31_373.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 78, step 37000: Generator loss: 2.2504563138484976, discriminator loss: 0.28022636586427685



![png](output_31_377.png)



![png](output_31_378.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 79, step 37500: Generator loss: 2.351982305526736, discriminator loss: 0.2753440611958504



![png](output_31_382.png)



![png](output_31_383.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 81, step 38000: Generator loss: 2.3168797879219043, discriminator loss: 0.2753816461563109



![png](output_31_389.png)



![png](output_31_390.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 82, step 38500: Generator loss: 2.2467314822673807, discriminator loss: 0.28615337669849356



![png](output_31_394.png)



![png](output_31_395.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 83, step 39000: Generator loss: 2.235527785778048, discriminator loss: 0.2860769817233085



![png](output_31_399.png)



![png](output_31_400.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 84, step 39500: Generator loss: 2.1566288626194003, discriminator loss: 0.2909596352875232



![png](output_31_404.png)



![png](output_31_405.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 85, step 40000: Generator loss: 2.2430428903102895, discriminator loss: 0.2728367485404015



![png](output_31_409.png)



![png](output_31_410.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 86, step 40500: Generator loss: 2.2214601290225975, discriminator loss: 0.28618702635169063



![png](output_31_414.png)



![png](output_31_415.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 87, step 41000: Generator loss: 2.291842972040176, discriminator loss: 0.2638240190148356



![png](output_31_419.png)



![png](output_31_420.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 88, step 41500: Generator loss: 2.08488910675049, discriminator loss: 0.31263120687007895



![png](output_31_424.png)



![png](output_31_425.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 89, step 42000: Generator loss: 2.163323547124863, discriminator loss: 0.2808202857375145



![png](output_31_429.png)



![png](output_31_430.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 90, step 42500: Generator loss: 2.1687695007324232, discriminator loss: 0.2945129549503325



![png](output_31_434.png)



![png](output_31_435.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 91, step 43000: Generator loss: 2.1472370216846466, discriminator loss: 0.2825999433994293



![png](output_31_439.png)



![png](output_31_440.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 92, step 43500: Generator loss: 2.196845761775972, discriminator loss: 0.27003445574641216



![png](output_31_444.png)



![png](output_31_445.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 93, step 44000: Generator loss: 2.1464265959262856, discriminator loss: 0.2839256367981432



![png](output_31_449.png)



![png](output_31_450.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 94, step 44500: Generator loss: 2.042095253944397, discriminator loss: 0.3042968268394471



![png](output_31_454.png)



![png](output_31_455.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 95, step 45000: Generator loss: 2.052191715002059, discriminator loss: 0.30367229959368697



![png](output_31_459.png)



![png](output_31_460.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 97, step 45500: Generator loss: 2.1427004506588, discriminator loss: 0.28819808900356275



![png](output_31_466.png)



![png](output_31_467.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 98, step 46000: Generator loss: 2.144683518409731, discriminator loss: 0.2863566573560237



![png](output_31_471.png)



![png](output_31_472.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 99, step 46500: Generator loss: 2.035548682689667, discriminator loss: 0.3073728800415992



![png](output_31_476.png)



![png](output_31_477.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 100, step 47000: Generator loss: 2.0681983010768894, discriminator loss: 0.29533911693096143



![png](output_31_481.png)



![png](output_31_482.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 101, step 47500: Generator loss: 2.0744604587554907, discriminator loss: 0.2984042511582375



![png](output_31_486.png)



![png](output_31_487.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 102, step 48000: Generator loss: 2.1228700540065764, discriminator loss: 0.28516901016235346



![png](output_31_491.png)



![png](output_31_492.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 103, step 48500: Generator loss: 2.0687562534809114, discriminator loss: 0.29754998362064317



![png](output_31_496.png)



![png](output_31_497.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 104, step 49000: Generator loss: 2.007203773021698, discriminator loss: 0.32414009213447587



![png](output_31_501.png)



![png](output_31_502.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 105, step 49500: Generator loss: 1.9600115432739276, discriminator loss: 0.3312604181468486



![png](output_31_506.png)



![png](output_31_507.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 106, step 50000: Generator loss: 2.1045495948791517, discriminator loss: 0.28588353142142303



![png](output_31_511.png)



![png](output_31_512.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 107, step 50500: Generator loss: 1.9982970063686363, discriminator loss: 0.310477391809225



![png](output_31_516.png)



![png](output_31_517.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 108, step 51000: Generator loss: 1.984551735401156, discriminator loss: 0.3148510728180409



![png](output_31_521.png)



![png](output_31_522.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 109, step 51500: Generator loss: 1.9500609459877003, discriminator loss: 0.32016990023851377



![png](output_31_526.png)



![png](output_31_527.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 110, step 52000: Generator loss: 1.9053118753433218, discriminator loss: 0.3393120501935483



![png](output_31_531.png)



![png](output_31_532.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 111, step 52500: Generator loss: 1.8652938218116757, discriminator loss: 0.3349670395851132



![png](output_31_536.png)



![png](output_31_537.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 113, step 53000: Generator loss: 1.9422638485431674, discriminator loss: 0.3270714125931262



![png](output_31_543.png)



![png](output_31_544.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 114, step 53500: Generator loss: 1.948348465442658, discriminator loss: 0.3202754564583302



![png](output_31_548.png)



![png](output_31_549.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 115, step 54000: Generator loss: 1.965461264610289, discriminator loss: 0.31608307981491085



![png](output_31_553.png)



![png](output_31_554.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 116, step 54500: Generator loss: 1.9133210005760182, discriminator loss: 0.3277061785757541



![png](output_31_558.png)



![png](output_31_559.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 117, step 55000: Generator loss: 1.8104710848331458, discriminator loss: 0.34054925486445436



![png](output_31_563.png)



![png](output_31_564.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 118, step 55500: Generator loss: 1.8033666846752174, discriminator loss: 0.34381638634204836



![png](output_31_568.png)



![png](output_31_569.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 119, step 56000: Generator loss: 1.7490416827201836, discriminator loss: 0.35326846802234674



![png](output_31_573.png)



![png](output_31_574.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 120, step 56500: Generator loss: 1.8908033576011656, discriminator loss: 0.32405790072679513



![png](output_31_578.png)



![png](output_31_579.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 121, step 57000: Generator loss: 1.8673703365325935, discriminator loss: 0.3471060887277127



![png](output_31_583.png)



![png](output_31_584.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 122, step 57500: Generator loss: 1.7666901807785034, discriminator loss: 0.3626599405407906



![png](output_31_588.png)



![png](output_31_589.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 123, step 58000: Generator loss: 1.7961963889598842, discriminator loss: 0.3486324425041675



![png](output_31_593.png)



![png](output_31_594.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 124, step 58500: Generator loss: 1.7691522834300994, discriminator loss: 0.3582459006905553



![png](output_31_598.png)



![png](output_31_599.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 125, step 59000: Generator loss: 1.8019114701747896, discriminator loss: 0.34446160745620735



![png](output_31_603.png)



![png](output_31_604.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 126, step 59500: Generator loss: 1.7323327555656425, discriminator loss: 0.3611531993150709



![png](output_31_608.png)



![png](output_31_609.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 127, step 60000: Generator loss: 1.8615225944519036, discriminator loss: 0.33485343462228784



![png](output_31_613.png)



![png](output_31_614.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 128, step 60500: Generator loss: 1.830304956436156, discriminator loss: 0.3459582275152204



![png](output_31_618.png)



![png](output_31_619.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 130, step 61000: Generator loss: 1.7557496132850652, discriminator loss: 0.35731643435359023



![png](output_31_625.png)



![png](output_31_626.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 131, step 61500: Generator loss: 1.7242148838043188, discriminator loss: 0.36557775849103946



![png](output_31_630.png)



![png](output_31_631.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 132, step 62000: Generator loss: 1.7648329553604138, discriminator loss: 0.3604291325807571



![png](output_31_635.png)



![png](output_31_636.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 133, step 62500: Generator loss: 1.5741851940155018, discriminator loss: 0.4016831576824193



![png](output_31_640.png)



![png](output_31_641.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 134, step 63000: Generator loss: 1.619758265972138, discriminator loss: 0.37987664651870695



![png](output_31_645.png)



![png](output_31_646.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 135, step 63500: Generator loss: 1.7429809739589692, discriminator loss: 0.3568324075341227



![png](output_31_650.png)



![png](output_31_651.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 136, step 64000: Generator loss: 1.6153542275428767, discriminator loss: 0.39530867260694535



![png](output_31_655.png)



![png](output_31_656.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 137, step 64500: Generator loss: 1.611156065464021, discriminator loss: 0.39318679714202887



![png](output_31_660.png)



![png](output_31_661.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 138, step 65000: Generator loss: 1.6157257976531976, discriminator loss: 0.3897301093935964



![png](output_31_665.png)



![png](output_31_666.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 139, step 65500: Generator loss: 1.5605379960536954, discriminator loss: 0.3982753610014916



![png](output_31_670.png)



![png](output_31_671.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 140, step 66000: Generator loss: 1.6611697266101826, discriminator loss: 0.3796099271178245



![png](output_31_675.png)



![png](output_31_676.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 141, step 66500: Generator loss: 1.6773000903129578, discriminator loss: 0.3589425099492072



![png](output_31_680.png)



![png](output_31_681.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 142, step 67000: Generator loss: 1.689275331735612, discriminator loss: 0.3731985949277878



![png](output_31_685.png)



![png](output_31_686.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 143, step 67500: Generator loss: 1.5095160903930667, discriminator loss: 0.41201399475336065



![png](output_31_690.png)



![png](output_31_691.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 144, step 68000: Generator loss: 1.4931480996608735, discriminator loss: 0.405467174172401



![png](output_31_695.png)



![png](output_31_696.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 146, step 68500: Generator loss: 1.6776898591518423, discriminator loss: 0.36413892102241513



![png](output_31_702.png)



![png](output_31_703.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 147, step 69000: Generator loss: 1.6895908913612374, discriminator loss: 0.37862336319685



![png](output_31_707.png)



![png](output_31_708.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 148, step 69500: Generator loss: 1.5637996480464924, discriminator loss: 0.4050430485010145



![png](output_31_712.png)



![png](output_31_713.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 149, step 70000: Generator loss: 1.5535046954154965, discriminator loss: 0.4024378095865254



![png](output_31_717.png)



![png](output_31_718.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 150, step 70500: Generator loss: 1.6374532074928287, discriminator loss: 0.37336305016279187



![png](output_31_722.png)



![png](output_31_723.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 151, step 71000: Generator loss: 1.6206455194950105, discriminator loss: 0.375690044462681



![png](output_31_727.png)



![png](output_31_728.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 152, step 71500: Generator loss: 1.706299899578094, discriminator loss: 0.3646494234204295



![png](output_31_732.png)



![png](output_31_733.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 153, step 72000: Generator loss: 1.6803401834964742, discriminator loss: 0.3672122831344599



![png](output_31_737.png)



![png](output_31_738.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 154, step 72500: Generator loss: 1.5999060077667238, discriminator loss: 0.3925746965408329



![png](output_31_742.png)



![png](output_31_743.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 155, step 73000: Generator loss: 1.5299538149833682, discriminator loss: 0.40185535681247747



![png](output_31_747.png)



![png](output_31_748.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 156, step 73500: Generator loss: 1.56719580578804, discriminator loss: 0.39544892323017117



![png](output_31_752.png)



![png](output_31_753.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 157, step 74000: Generator loss: 1.5263997211456295, discriminator loss: 0.39516326063871365



![png](output_31_757.png)



![png](output_31_758.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 158, step 74500: Generator loss: 1.5980213441848756, discriminator loss: 0.38499842464923895



![png](output_31_762.png)



![png](output_31_763.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 159, step 75000: Generator loss: 1.5030803594589246, discriminator loss: 0.40497418963909154



![png](output_31_767.png)



![png](output_31_768.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 160, step 75500: Generator loss: 1.5686077790260318, discriminator loss: 0.3976942868232727



![png](output_31_772.png)



![png](output_31_773.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 162, step 76000: Generator loss: 1.5185226669311531, discriminator loss: 0.4155429173111912



![png](output_31_779.png)



![png](output_31_780.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 163, step 76500: Generator loss: 1.5126711962223043, discriminator loss: 0.4088978594541552



![png](output_31_784.png)



![png](output_31_785.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 164, step 77000: Generator loss: 1.4737814288139333, discriminator loss: 0.4200247518420217



![png](output_31_789.png)



![png](output_31_790.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 165, step 77500: Generator loss: 1.5350538625717158, discriminator loss: 0.39222454440593674



![png](output_31_794.png)



![png](output_31_795.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 166, step 78000: Generator loss: 1.550416028499603, discriminator loss: 0.40166401928663303



![png](output_31_799.png)



![png](output_31_800.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 167, step 78500: Generator loss: 1.5270967578887935, discriminator loss: 0.392940763890743



![png](output_31_804.png)



![png](output_31_805.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 168, step 79000: Generator loss: 1.4253702197074873, discriminator loss: 0.44024469989538223



![png](output_31_809.png)



![png](output_31_810.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 169, step 79500: Generator loss: 1.4477621624469763, discriminator loss: 0.419251076340675



![png](output_31_814.png)



![png](output_31_815.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 170, step 80000: Generator loss: 1.516675207138063, discriminator loss: 0.40833766609430333



![png](output_31_819.png)



![png](output_31_820.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 171, step 80500: Generator loss: 1.500562129974366, discriminator loss: 0.41940853732824346



![png](output_31_824.png)



![png](output_31_825.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 172, step 81000: Generator loss: 1.468521190166474, discriminator loss: 0.4264453019499781



![png](output_31_829.png)



![png](output_31_830.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 173, step 81500: Generator loss: 1.4230723352432242, discriminator loss: 0.43732614028453876



![png](output_31_834.png)



![png](output_31_835.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 174, step 82000: Generator loss: 1.525965398550032, discriminator loss: 0.40918993967771516



![png](output_31_839.png)



![png](output_31_840.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 175, step 82500: Generator loss: 1.4351144907474527, discriminator loss: 0.43674448114633563



![png](output_31_844.png)



![png](output_31_845.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 176, step 83000: Generator loss: 1.3225678825378404, discriminator loss: 0.46930172842740997



![png](output_31_849.png)



![png](output_31_850.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 178, step 83500: Generator loss: 1.460192065000534, discriminator loss: 0.415296378314495



![png](output_31_856.png)



![png](output_31_857.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 179, step 84000: Generator loss: 1.420117113590239, discriminator loss: 0.4377719452977183



![png](output_31_861.png)



![png](output_31_862.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 180, step 84500: Generator loss: 1.3952573845386502, discriminator loss: 0.44531083971261964



![png](output_31_866.png)



![png](output_31_867.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 181, step 85000: Generator loss: 1.3418873803615559, discriminator loss: 0.4549565444588662



![png](output_31_871.png)



![png](output_31_872.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 182, step 85500: Generator loss: 1.2603452231883994, discriminator loss: 0.49649087184667595



![png](output_31_876.png)



![png](output_31_877.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 183, step 86000: Generator loss: 1.3346679062843325, discriminator loss: 0.4612312952280047



![png](output_31_881.png)



![png](output_31_882.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 184, step 86500: Generator loss: 1.38446350288391, discriminator loss: 0.44942715424299234



![png](output_31_886.png)



![png](output_31_887.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 185, step 87000: Generator loss: 1.3292756817340854, discriminator loss: 0.4665924875140192



![png](output_31_891.png)



![png](output_31_892.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 186, step 87500: Generator loss: 1.3090171620845799, discriminator loss: 0.4673720863461492



![png](output_31_896.png)



![png](output_31_897.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 187, step 88000: Generator loss: 1.3527162175178526, discriminator loss: 0.45077835565805424



![png](output_31_901.png)



![png](output_31_902.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 188, step 88500: Generator loss: 1.4191978824138638, discriminator loss: 0.4278976275920867



![png](output_31_906.png)



![png](output_31_907.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 189, step 89000: Generator loss: 1.321913980007171, discriminator loss: 0.4636027719974515



![png](output_31_911.png)



![png](output_31_912.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 190, step 89500: Generator loss: 1.3083437881469726, discriminator loss: 0.46014183825254423



![png](output_31_916.png)



![png](output_31_917.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 191, step 90000: Generator loss: 1.3605687420368198, discriminator loss: 0.4443725310564038



![png](output_31_921.png)



![png](output_31_922.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 192, step 90500: Generator loss: 1.407956863164902, discriminator loss: 0.418681949555873



![png](output_31_926.png)



![png](output_31_927.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 194, step 91000: Generator loss: 1.3578461925983436, discriminator loss: 0.4506306265592577



![png](output_31_933.png)



![png](output_31_934.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 195, step 91500: Generator loss: 1.3812076690196984, discriminator loss: 0.438279487252235



![png](output_31_938.png)



![png](output_31_939.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 196, step 92000: Generator loss: 1.2666883714199075, discriminator loss: 0.4846302587389945



![png](output_31_943.png)



![png](output_31_944.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 197, step 92500: Generator loss: 1.2974200048446651, discriminator loss: 0.4650121510028841



![png](output_31_948.png)



![png](output_31_949.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 198, step 93000: Generator loss: 1.2808977520465854, discriminator loss: 0.4623664264678952



![png](output_31_953.png)



![png](output_31_954.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 199, step 93500: Generator loss: 1.2486372106075276, discriminator loss: 0.4813919951915738



![png](output_31_958.png)



![png](output_31_959.png)


    



```python

```
