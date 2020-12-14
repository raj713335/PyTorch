# Wasserstein GAN with Gradient Penalty (WGAN-GP)

### Goals
In this notebook, you're going to build a Wasserstein GAN with Gradient Penalty (WGAN-GP) that solves some of the stability issues with the GANs that you have been using up until this point. Specifically, you'll use a special kind of loss function known as the W-loss, where W stands for Wasserstein, and gradient penalties to prevent mode collapse.

*Fun Fact: Wasserstein is named after a mathematician at Penn State, Leonid Vaseršteĭn. You'll see it abbreviated to W (e.g. WGAN, W-loss, W-distance).*

### Learning Objectives
1.   Get hands-on experience building a more stable GAN: Wasserstein GAN with Gradient Penalty (WGAN-GP).
2.   Train the more advanced WGAN-GP model.



## Generator and Critic

You will begin by importing some useful packages, defining visualization functions, building the generator, and building the critic. Since the changes for WGAN-GP are done to the loss function during training, you can simply reuse your previous GAN code for the generator and critic class. Remember that in WGAN-GP, you no longer use a discriminator that classifies fake and real as 0 and 1 but rather a critic that scores images with real numbers.

#### Packages and Visualizations


```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes, 
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook
```

#### Generator and Noise


```python
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)
```

#### Critic


```python
class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)
```

## Training Initializations
Now you can start putting it all together.
As usual, you will start by setting the parameters:
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   beta_1, beta_2: the momentum terms
  *   c_lambda: weight of the gradient penalty
  *   crit_repeats: number of times to update the critic per generator update - there are more details about this in the *Putting It All Together* section
  *   device: the device type

You will also load and transform the MNIST dataset to tensors.





```python
n_epochs = 100
z_dim = 64
display_step = 50
batch_size = 128
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cuda'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)
```

Then, you can initialize your generator, critic, and optimizers.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device) 
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

```

## Gradient Penalty
Calculating the gradient penalty can be broken into two functions: (1) compute the gradient with respect to the images and (2) compute the gradient penalty given the gradient.

You can start by getting the gradient. The gradient is computed by first creating a mixed image. This is done by weighing the fake and real image using epsilon and then adding them together. Once you have the intermediate image, you can get the critic's output on the image. Finally, you compute the gradient of the critic score's on the mixed images (output) with respect to the pixels of the mixed images (input). You will need to fill in the code to get the gradient wherever you see *None*. There is a test function in the next block for you to test your solution.


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gradient
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

```


```python
# UNIT TEST
# DO NOT MODIFY THIS
def test_get_gradient(image_shape):
    real = torch.randn(*image_shape, device=device) + 1
    fake = torch.randn(*image_shape, device=device) - 1
    epsilon_shape = [1 for _ in image_shape]
    epsilon_shape[0] = image_shape[0]
    epsilon = torch.rand(epsilon_shape, device=device).requires_grad_()
    gradient = get_gradient(crit, real, fake, epsilon)
    assert tuple(gradient.shape) == image_shape
    assert gradient.max() > 0
    assert gradient.min() < 0
    return gradient

gradient = test_get_gradient((256, 1, 28, 28))
print("Success!")
```

    Success!


The second function you need to complete is to compute the gradient penalty given the gradient. First, you calculate the magnitude of each image's gradient. The magnitude of a gradient is also called the norm. Then, you calculate the penalty by squaring the distance between each magnitude and the ideal norm of 1 and taking the mean of all the squared distances.

Again, you will need to fill in the code wherever you see *None*. There are hints below that you can view if you need help and there is a test function in the next block for you to test your solution.

<details>

<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">gradient_penalty</font></code></b>
</font>
</summary>


1.   Make sure you take the mean at the end.
2.   Note that the magnitude of each gradient has already been calculated for you.

</details>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: gradient_penalty
def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm - 1)**2)
    #### END CODE HERE ####
    return penalty
```


```python
# UNIT TEST
def test_gradient_penalty(image_shape):
    bad_gradient = torch.zeros(*image_shape)
    bad_gradient_penalty = gradient_penalty(bad_gradient)
    assert torch.isclose(bad_gradient_penalty, torch.tensor(1.))

    image_size = torch.prod(torch.Tensor(image_shape[1:]))
    good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)
    good_gradient_penalty = gradient_penalty(good_gradient)
    assert torch.isclose(good_gradient_penalty, torch.tensor(0.))

    random_gradient = test_get_gradient(image_shape)
    random_gradient_penalty = gradient_penalty(random_gradient)
    assert torch.abs(random_gradient_penalty - 1) < 0.1

test_gradient_penalty((256, 1, 28, 28))
print("Success!")
```

    Success!


## Losses
Next, you need to calculate the loss for the generator and the critic.

For the generator, the loss is calculated by maximizing the critic's prediction on the generator's fake images. The argument has the scores for all fake images in the batch, but you will use the mean of them.

There are optional hints below and a test function in the next block for you to test your solution.

<details><summary><font size="3" color="green"><b>Optional hints for <code><font size="4">get_gen_loss</font></code></b></font></summary>

1. This can be written in one line.
2. This is the negative of the mean of the critic's scores.

</details>


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    #### START CODE HERE ####
    gen_loss = -1. * torch.mean(crit_fake_pred)
    #### END CODE HERE ####
    return gen_loss
```


```python
# UNIT TEST
assert torch.isclose(
    get_gen_loss(torch.tensor(1.)), torch.tensor(-1.0)
)

assert torch.isclose(
    get_gen_loss(torch.rand(10000)), torch.tensor(-0.5), 0.05
)

print("Success!")
```

    Success!


For the critic, the loss is calculated by maximizing the distance between the critic's predictions on the real images and the predictions on the fake images while also adding a gradient penalty. The gradient penalty is weighed according to lambda. The arguments are the scores for all the images in the batch, and you will use the mean of them.

There are hints below if you get stuck and a test function in the next block for you to test your solution.

<details><summary><font size="3" color="green"><b>Optional hints for <code><font size="4">get_crit_loss</font></code></b></font></summary>

1. The higher the mean fake score, the higher the critic's loss is.
2. What does this suggest about the mean real score?
3. The higher the gradient penalty, the higher the critic's loss is, proportional to lambda.


</details>



```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_crit_loss
def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    #### START CODE HERE ####
    crit_loss =  torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    #### END CODE HERE ####
    return crit_loss
```


```python
# UNIT TEST
assert torch.isclose(
    get_crit_loss(torch.tensor(1.), torch.tensor(2.), torch.tensor(3.), 0.1),
    torch.tensor(-0.7)
)
assert torch.isclose(
    get_crit_loss(torch.tensor(20.), torch.tensor(-20.), torch.tensor(2.), 10),
    torch.tensor(60.)
)

print("Success!")
```

    Success!


## Putting It All Together
Before you put everything together, there are a few things to note.
1.   Even on GPU, the **training will run more slowly** than previous labs because the gradient penalty requires you to compute the gradient of a gradient -- this means potentially a few minutes per epoch! For best results, run this for as long as you can while on GPU.
2.   One important difference from earlier versions is that you will **update the critic multiple times** every time you update the generator This helps prevent the generator from overpowering the critic. Sometimes, you might see the reverse, with the generator updated more times than the critic. This depends on architectural (e.g. the depth and width of the network) and algorithmic choices (e.g. which loss you're using). 
3.   WGAN-GP isn't necessarily meant to improve overall performance of a GAN, but just **increases stability** and avoids mode collapse. In general, a WGAN will be able to train in a much more stable way than the vanilla DCGAN from last assignment, though it will generally run a bit slower. You should also be able to train your model for more epochs without it collapsing.


<!-- Once again, be warned that this runs very slowly on a CPU. One way to run this more quickly is to download the .ipynb and upload it to Google Drive, then open it with Google Colab and make the runtime type GPU and replace
`device = "cpu"`
with
`device = "cuda"`
and make sure that your `get_noise` function uses the right device.  -->

Here is a snapshot of what your WGAN-GP outputs should resemble:
![MNIST Digits Progression](MNIST_WGAN_Progression.png)


```python
import matplotlib.pyplot as plt

cur_step = 0
generator_losses = []
critic_losses = []
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)
        
        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        # Update the weights
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1

```


    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 50: Generator loss: -0.08921420384198427, critic loss: 1.8831281597912308



![png](output_26_2.png)



![png](output_26_3.png)



![png](output_26_4.png)


    Step 100: Generator loss: 0.9910049936175347, critic loss: -2.0206484434604643



![png](output_26_6.png)



![png](output_26_7.png)



![png](output_26_8.png)


    Step 150: Generator loss: 3.046200919151306, critic loss: -11.298576456069943



![png](output_26_10.png)



![png](output_26_11.png)



![png](output_26_12.png)


    Step 200: Generator loss: 1.9648561203479766, critic loss: -28.713224487304682



![png](output_26_14.png)



![png](output_26_15.png)



![png](output_26_16.png)


    Step 250: Generator loss: 1.615120728611946, critic loss: -55.80875505065917



![png](output_26_18.png)



![png](output_26_19.png)



![png](output_26_20.png)


    Step 300: Generator loss: 2.8086279916763304, critic loss: -90.0936875



![png](output_26_22.png)



![png](output_26_23.png)



![png](output_26_24.png)


    Step 350: Generator loss: 6.6634070634841915, critic loss: -131.57128985595705



![png](output_26_26.png)



![png](output_26_27.png)



![png](output_26_28.png)


    Step 400: Generator loss: 11.841998767852782, critic loss: -179.63054962158202



![png](output_26_30.png)



![png](output_26_31.png)



![png](output_26_32.png)


    Step 450: Generator loss: 15.344743099212646, critic loss: -226.48242974853514



![png](output_26_34.png)



![png](output_26_35.png)



![png](output_26_36.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 500: Generator loss: 19.01122001647949, critic loss: -279.23986126708985



![png](output_26_40.png)



![png](output_26_41.png)



![png](output_26_42.png)


    Step 550: Generator loss: 22.643907604217528, critic loss: -329.1926704711914



![png](output_26_44.png)



![png](output_26_45.png)



![png](output_26_46.png)


    Step 600: Generator loss: 10.9468537068367, critic loss: -337.03613214111317



![png](output_26_48.png)



![png](output_26_49.png)



![png](output_26_50.png)


    Step 650: Generator loss: 0.29342069387435915, critic loss: -282.8964716491699



![png](output_26_52.png)



![png](output_26_53.png)



![png](output_26_54.png)


    Step 700: Generator loss: -10.983209661841393, critic loss: -233.03167108154304



![png](output_26_56.png)



![png](output_26_57.png)



![png](output_26_58.png)


    Step 750: Generator loss: -10.81807785987854, critic loss: -204.6148975830078



![png](output_26_60.png)



![png](output_26_61.png)



![png](output_26_62.png)


    Step 800: Generator loss: 0.59345518887043, critic loss: -123.35655097198482



![png](output_26_64.png)



![png](output_26_65.png)



![png](output_26_66.png)


    Step 850: Generator loss: -12.356699594259261, critic loss: -139.1475469970703



![png](output_26_68.png)



![png](output_26_69.png)



![png](output_26_70.png)


    Step 900: Generator loss: -4.773474552631378, critic loss: -43.391711318969726



![png](output_26_72.png)



![png](output_26_73.png)



![png](output_26_74.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 950: Generator loss: -39.048114286661146, critic loss: -116.28293243408201



![png](output_26_78.png)



![png](output_26_79.png)



![png](output_26_80.png)


    Step 1000: Generator loss: -36.18996314048767, critic loss: -94.15679683685303



![png](output_26_82.png)



![png](output_26_83.png)



![png](output_26_84.png)


    Step 1050: Generator loss: -24.65136296749115, critic loss: -89.51137709045412



![png](output_26_86.png)



![png](output_26_87.png)



![png](output_26_88.png)


    Step 1100: Generator loss: -32.54520573973656, critic loss: -39.82132741546629



![png](output_26_90.png)



![png](output_26_91.png)



![png](output_26_92.png)


    Step 1150: Generator loss: 2.7156317329406736, critic loss: -5.445600322723387



![png](output_26_94.png)



![png](output_26_95.png)



![png](output_26_96.png)


    Step 1200: Generator loss: -36.49390377521515, critic loss: -65.9989951324463



![png](output_26_98.png)



![png](output_26_99.png)



![png](output_26_100.png)


    Step 1250: Generator loss: -6.092022314071655, critic loss: 1.9601769561767506



![png](output_26_102.png)



![png](output_26_103.png)



![png](output_26_104.png)


    Step 1300: Generator loss: 17.928531188964843, critic loss: 26.649888221740724



![png](output_26_106.png)



![png](output_26_107.png)



![png](output_26_108.png)


    Step 1350: Generator loss: 20.14070873260498, critic loss: 39.757784774780276



![png](output_26_110.png)



![png](output_26_111.png)



![png](output_26_112.png)


    Step 1400: Generator loss: 20.83430950164795, critic loss: 27.006047061920164



![png](output_26_114.png)



![png](output_26_115.png)



![png](output_26_116.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 1450: Generator loss: 22.07602195739746, critic loss: 20.33825241470337



![png](output_26_120.png)



![png](output_26_121.png)



![png](output_26_122.png)


    Step 1500: Generator loss: 22.679368324279785, critic loss: 32.26377305603028



![png](output_26_124.png)



![png](output_26_125.png)



![png](output_26_126.png)


    Step 1550: Generator loss: 21.543437423706056, critic loss: 51.227098556518555



![png](output_26_128.png)



![png](output_26_129.png)



![png](output_26_130.png)


    Step 1600: Generator loss: 18.704231147766112, critic loss: 39.15597252655029



![png](output_26_132.png)



![png](output_26_133.png)



![png](output_26_134.png)


    Step 1650: Generator loss: 17.865181427001954, critic loss: 21.55395868301391



![png](output_26_136.png)



![png](output_26_137.png)



![png](output_26_138.png)


    Step 1700: Generator loss: 18.9741646194458, critic loss: 11.251179721832274



![png](output_26_140.png)



![png](output_26_141.png)



![png](output_26_142.png)


    Step 1750: Generator loss: 19.832909126281738, critic loss: 6.178790395736696



![png](output_26_144.png)



![png](output_26_145.png)



![png](output_26_146.png)


    Step 1800: Generator loss: 19.7320858001709, critic loss: 2.2933278121948244



![png](output_26_148.png)



![png](output_26_149.png)



![png](output_26_150.png)


    Step 1850: Generator loss: 18.630216789245605, critic loss: -0.05408877468109131



![png](output_26_152.png)



![png](output_26_153.png)



![png](output_26_154.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 1900: Generator loss: 17.853739738464355, critic loss: -1.9321029586791993



![png](output_26_158.png)



![png](output_26_159.png)



![png](output_26_160.png)


    Step 1950: Generator loss: 17.81026798248291, critic loss: -3.6949886760711674



![png](output_26_162.png)



![png](output_26_163.png)



![png](output_26_164.png)


    Step 2000: Generator loss: 17.761905784606935, critic loss: -5.694157825469971



![png](output_26_166.png)



![png](output_26_167.png)



![png](output_26_168.png)


    Step 2050: Generator loss: 19.012782211303712, critic loss: -7.957532529830933



![png](output_26_170.png)



![png](output_26_171.png)



![png](output_26_172.png)


    Step 2100: Generator loss: 20.220277938842774, critic loss: -8.964875978469847



![png](output_26_174.png)



![png](output_26_175.png)



![png](output_26_176.png)


    Step 2150: Generator loss: 21.793281593322753, critic loss: -9.589056808471678



![png](output_26_178.png)



![png](output_26_179.png)



![png](output_26_180.png)


    Step 2200: Generator loss: 23.777834320068358, critic loss: -11.204513425827026



![png](output_26_182.png)



![png](output_26_183.png)



![png](output_26_184.png)


    Step 2250: Generator loss: 25.60374683380127, critic loss: -13.55532264137268



![png](output_26_186.png)



![png](output_26_187.png)



![png](output_26_188.png)


    Step 2300: Generator loss: 26.666469345092775, critic loss: -15.503362087249755



![png](output_26_190.png)



![png](output_26_191.png)



![png](output_26_192.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 2350: Generator loss: 28.574446182250977, critic loss: -17.040301713943485



![png](output_26_196.png)



![png](output_26_197.png)



![png](output_26_198.png)


    Step 2400: Generator loss: 29.376011466979982, critic loss: -17.610908866882323



![png](output_26_200.png)



![png](output_26_201.png)



![png](output_26_202.png)


    Step 2450: Generator loss: 29.95771415710449, critic loss: -19.20072966766357



![png](output_26_204.png)



![png](output_26_205.png)



![png](output_26_206.png)


    Step 2500: Generator loss: 32.90923027038574, critic loss: -20.993191062927245



![png](output_26_208.png)



![png](output_26_209.png)



![png](output_26_210.png)


    Step 2550: Generator loss: 31.34291130065918, critic loss: -17.45489587497711



![png](output_26_212.png)



![png](output_26_213.png)



![png](output_26_214.png)


    Step 2600: Generator loss: 32.966843109130856, critic loss: -21.94545089149475



![png](output_26_216.png)



![png](output_26_217.png)



![png](output_26_218.png)


    Step 2650: Generator loss: 33.40266414642334, critic loss: -22.9198738117218



![png](output_26_220.png)



![png](output_26_221.png)



![png](output_26_222.png)


    Step 2700: Generator loss: 34.30109714508057, critic loss: -23.066707313537613



![png](output_26_224.png)



![png](output_26_225.png)



![png](output_26_226.png)


    Step 2750: Generator loss: 34.61113380432129, critic loss: -17.36830479097366



![png](output_26_228.png)



![png](output_26_229.png)



![png](output_26_230.png)


    Step 2800: Generator loss: 33.29627773284912, critic loss: -2.576256610870362



![png](output_26_232.png)



![png](output_26_233.png)



![png](output_26_234.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 2850: Generator loss: 29.558516998291015, critic loss: -3.051739415168764



![png](output_26_238.png)



![png](output_26_239.png)



![png](output_26_240.png)


    Step 2900: Generator loss: 26.49561180114746, critic loss: -7.578926792144776



![png](output_26_242.png)



![png](output_26_243.png)



![png](output_26_244.png)


    Step 2950: Generator loss: 29.630672569274903, critic loss: -17.352051138401027



![png](output_26_246.png)



![png](output_26_247.png)



![png](output_26_248.png)


    Step 3000: Generator loss: 35.99958890914917, critic loss: -3.8786504755020137



![png](output_26_250.png)



![png](output_26_251.png)



![png](output_26_252.png)


    Step 3050: Generator loss: 32.82471012115479, critic loss: -15.410893178939823



![png](output_26_254.png)



![png](output_26_255.png)



![png](output_26_256.png)


    Step 3100: Generator loss: 35.89292545318604, critic loss: -11.659702900886542



![png](output_26_258.png)



![png](output_26_259.png)



![png](output_26_260.png)


    Step 3150: Generator loss: 32.24928127288818, critic loss: -21.518912685394287



![png](output_26_262.png)



![png](output_26_263.png)



![png](output_26_264.png)


    Step 3200: Generator loss: 35.46997859954834, critic loss: -22.931143020629882



![png](output_26_266.png)



![png](output_26_267.png)



![png](output_26_268.png)


    Step 3250: Generator loss: 37.7632731628418, critic loss: -16.99604215812683



![png](output_26_270.png)



![png](output_26_271.png)



![png](output_26_272.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 3300: Generator loss: 38.334139556884764, critic loss: -14.334082129478452



![png](output_26_276.png)



![png](output_26_277.png)



![png](output_26_278.png)


    Step 3350: Generator loss: 37.13550666809082, critic loss: -22.36182120037079



![png](output_26_280.png)



![png](output_26_281.png)



![png](output_26_282.png)


    Step 3400: Generator loss: 38.833131294250485, critic loss: -23.527738759994506



![png](output_26_284.png)



![png](output_26_285.png)



![png](output_26_286.png)


    Step 3450: Generator loss: 40.899076232910154, critic loss: -13.445384818077084



![png](output_26_288.png)



![png](output_26_289.png)



![png](output_26_290.png)


    Step 3500: Generator loss: 40.24483642578125, critic loss: -5.536909059524536



![png](output_26_292.png)



![png](output_26_293.png)



![png](output_26_294.png)


    Step 3550: Generator loss: 34.13941074371338, critic loss: -21.115789493560797



![png](output_26_296.png)



![png](output_26_297.png)



![png](output_26_298.png)


    Step 3600: Generator loss: 39.02042205810547, critic loss: -16.819884380340575



![png](output_26_300.png)



![png](output_26_301.png)



![png](output_26_302.png)


    Step 3650: Generator loss: 42.37631980895996, critic loss: -15.900618944168095



![png](output_26_304.png)



![png](output_26_305.png)



![png](output_26_306.png)


    Step 3700: Generator loss: 44.59690788269043, critic loss: -10.957927131652834



![png](output_26_308.png)



![png](output_26_309.png)



![png](output_26_310.png)


    Step 3750: Generator loss: 40.69034877777099, critic loss: -20.309734083175663



![png](output_26_312.png)



![png](output_26_313.png)



![png](output_26_314.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 3800: Generator loss: 47.231308479309085, critic loss: -6.303932723999023



![png](output_26_318.png)



![png](output_26_319.png)



![png](output_26_320.png)


    Step 3850: Generator loss: 43.24249839782715, critic loss: -16.592897315025333



![png](output_26_322.png)



![png](output_26_323.png)



![png](output_26_324.png)


    Step 3900: Generator loss: 43.93731185913086, critic loss: -20.33991699409485



![png](output_26_326.png)



![png](output_26_327.png)



![png](output_26_328.png)


    Step 3950: Generator loss: 47.979606552124025, critic loss: -9.223579836845397



![png](output_26_330.png)



![png](output_26_331.png)



![png](output_26_332.png)


    Step 4000: Generator loss: 46.665524368286135, critic loss: -7.325505867004393



![png](output_26_334.png)



![png](output_26_335.png)



![png](output_26_336.png)


    Step 4050: Generator loss: 43.50457946777344, critic loss: -19.599961451530458



![png](output_26_338.png)



![png](output_26_339.png)



![png](output_26_340.png)


    Step 4100: Generator loss: 48.575121307373045, critic loss: -12.0803477487564



![png](output_26_342.png)



![png](output_26_343.png)



![png](output_26_344.png)


    Step 4150: Generator loss: 50.74765518188477, critic loss: -12.139481350898743



![png](output_26_346.png)



![png](output_26_347.png)



![png](output_26_348.png)


    Step 4200: Generator loss: 48.069759902954104, critic loss: -20.85440789413452



![png](output_26_350.png)



![png](output_26_351.png)



![png](output_26_352.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 4250: Generator loss: 55.02804941177368, critic loss: -10.325732482910155



![png](output_26_356.png)



![png](output_26_357.png)



![png](output_26_358.png)


    Step 4300: Generator loss: 49.63911407470703, critic loss: -18.507485795021058



![png](output_26_360.png)



![png](output_26_361.png)



![png](output_26_362.png)


    Step 4350: Generator loss: 51.40134284973144, critic loss: -20.766491333961486



![png](output_26_364.png)



![png](output_26_365.png)



![png](output_26_366.png)


    Step 4400: Generator loss: 53.89939666748047, critic loss: -18.530056619644167



![png](output_26_368.png)



![png](output_26_369.png)



![png](output_26_370.png)


    Step 4450: Generator loss: 61.22015647888183, critic loss: -7.963024959564209



![png](output_26_372.png)



![png](output_26_373.png)



![png](output_26_374.png)


    Step 4500: Generator loss: 63.189253387451174, critic loss: -3.5767962894439718



![png](output_26_376.png)



![png](output_26_377.png)



![png](output_26_378.png)


    Step 4550: Generator loss: 53.97570510864258, critic loss: -13.213317111015323



![png](output_26_380.png)



![png](output_26_381.png)



![png](output_26_382.png)


    Step 4600: Generator loss: 54.18066619873047, critic loss: -18.101094491958612



![png](output_26_384.png)



![png](output_26_385.png)



![png](output_26_386.png)


    Step 4650: Generator loss: 63.074241638183594, critic loss: -9.517001467704775



![png](output_26_388.png)



![png](output_26_389.png)



![png](output_26_390.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 4700: Generator loss: 65.33966354370118, critic loss: -5.796388601303102



![png](output_26_394.png)



![png](output_26_395.png)



![png](output_26_396.png)


    Step 4750: Generator loss: 57.94232627868652, critic loss: -19.104323853015902



![png](output_26_398.png)



![png](output_26_399.png)



![png](output_26_400.png)


    Step 4800: Generator loss: 61.99587738037109, critic loss: -14.573363418340687



![png](output_26_402.png)



![png](output_26_403.png)



![png](output_26_404.png)


    Step 4850: Generator loss: 62.91675193786621, critic loss: -18.941992817878724



![png](output_26_406.png)



![png](output_26_407.png)



![png](output_26_408.png)


    Step 4900: Generator loss: 62.570198440551756, critic loss: -18.47068563747406



![png](output_26_410.png)



![png](output_26_411.png)



![png](output_26_412.png)


    Step 4950: Generator loss: 66.52637870788574, critic loss: -18.094282247066502



![png](output_26_414.png)



![png](output_26_415.png)



![png](output_26_416.png)


    Step 5000: Generator loss: 66.16911491394043, critic loss: -9.031688045501712



![png](output_26_418.png)



![png](output_26_419.png)



![png](output_26_420.png)


    Step 5050: Generator loss: 63.82594581604004, critic loss: -17.67154801368714



![png](output_26_422.png)



![png](output_26_423.png)



![png](output_26_424.png)


    Step 5100: Generator loss: 66.34983924865723, critic loss: -17.168054577827455



![png](output_26_426.png)



![png](output_26_427.png)



![png](output_26_428.png)


    Step 5150: Generator loss: 66.78400695800781, critic loss: -18.803219132423404



![png](output_26_430.png)



![png](output_26_431.png)



![png](output_26_432.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 5200: Generator loss: 67.49511436462403, critic loss: -15.756955276489258



![png](output_26_436.png)



![png](output_26_437.png)



![png](output_26_438.png)


    Step 5250: Generator loss: 72.89446716308593, critic loss: -10.822006483078



![png](output_26_440.png)



![png](output_26_441.png)



![png](output_26_442.png)


    Step 5300: Generator loss: 70.03119827270508, critic loss: -17.11750562953949



![png](output_26_444.png)



![png](output_26_445.png)



![png](output_26_446.png)


    Step 5350: Generator loss: 70.34807327270508, critic loss: -15.545968837738037



![png](output_26_448.png)



![png](output_26_449.png)



![png](output_26_450.png)


    Step 5400: Generator loss: 74.98057540893555, critic loss: -15.267065300941464



![png](output_26_452.png)



![png](output_26_453.png)



![png](output_26_454.png)


    Step 5450: Generator loss: 78.19376853942872, critic loss: -6.081193484306335



![png](output_26_456.png)



![png](output_26_457.png)



![png](output_26_458.png)


    Step 5500: Generator loss: 73.40428443908691, critic loss: -16.29325572872161



![png](output_26_460.png)



![png](output_26_461.png)



![png](output_26_462.png)


    Step 5550: Generator loss: 72.58103828430175, critic loss: -17.578088739871976



![png](output_26_464.png)



![png](output_26_465.png)



![png](output_26_466.png)


    Step 5600: Generator loss: 71.61492507934571, critic loss: -20.735977544784546



![png](output_26_468.png)



![png](output_26_469.png)



![png](output_26_470.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 5650: Generator loss: 82.64605766296387, critic loss: -4.906936118602752



![png](output_26_474.png)



![png](output_26_475.png)



![png](output_26_476.png)


    Step 5700: Generator loss: 83.23922592163086, critic loss: -3.710829990386963



![png](output_26_478.png)



![png](output_26_479.png)



![png](output_26_480.png)


    Step 5750: Generator loss: 77.58529884338378, critic loss: -8.36132246398926



![png](output_26_482.png)



![png](output_26_483.png)



![png](output_26_484.png)


    Step 5800: Generator loss: 80.52659301757812, critic loss: -12.004045170784



![png](output_26_486.png)



![png](output_26_487.png)



![png](output_26_488.png)


    Step 5850: Generator loss: 76.7947930908203, critic loss: -16.625511860847475



![png](output_26_490.png)



![png](output_26_491.png)



![png](output_26_492.png)


    Step 5900: Generator loss: 82.44211212158203, critic loss: -9.15983170700073



![png](output_26_494.png)



![png](output_26_495.png)



![png](output_26_496.png)


    Step 5950: Generator loss: 79.4135498046875, critic loss: -13.162193954467773



![png](output_26_498.png)



![png](output_26_499.png)



![png](output_26_500.png)


    Step 6000: Generator loss: 78.07337539672852, critic loss: -17.067728280067445



![png](output_26_502.png)



![png](output_26_503.png)



![png](output_26_504.png)


    Step 6050: Generator loss: 88.3516000366211, critic loss: 1.8829628233909608



![png](output_26_506.png)



![png](output_26_507.png)



![png](output_26_508.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 6100: Generator loss: 87.18871002197265, critic loss: -2.4111183948516848



![png](output_26_512.png)



![png](output_26_513.png)



![png](output_26_514.png)


    Step 6150: Generator loss: 85.98255966186524, critic loss: -2.65972613620758



![png](output_26_516.png)



![png](output_26_517.png)



![png](output_26_518.png)


    Step 6200: Generator loss: 84.67106185913086, critic loss: -2.9461547098159797



![png](output_26_520.png)



![png](output_26_521.png)



![png](output_26_522.png)


    Step 6250: Generator loss: 83.24468124389648, critic loss: -3.2197943811416616



![png](output_26_524.png)



![png](output_26_525.png)



![png](output_26_526.png)


    Step 6300: Generator loss: 79.76938972473144, critic loss: -4.6463210043907175



![png](output_26_528.png)



![png](output_26_529.png)



![png](output_26_530.png)


    Step 6350: Generator loss: 84.55428466796874, critic loss: -4.906443878173828



![png](output_26_532.png)



![png](output_26_533.png)



![png](output_26_534.png)


    Step 6400: Generator loss: 77.9591081237793, critic loss: -15.462328550338743



![png](output_26_536.png)



![png](output_26_537.png)



![png](output_26_538.png)


    Step 6450: Generator loss: 80.21318756103516, critic loss: -12.29338437652588



![png](output_26_540.png)



![png](output_26_541.png)



![png](output_26_542.png)


    Step 6500: Generator loss: 79.44552429199219, critic loss: -16.175099219322203



![png](output_26_544.png)



![png](output_26_545.png)



![png](output_26_546.png)


    Step 6550: Generator loss: 81.64179595947266, critic loss: -13.96830569505691



![png](output_26_548.png)



![png](output_26_549.png)



![png](output_26_550.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 6600: Generator loss: 80.00523315429687, critic loss: -17.837730355262753



![png](output_26_554.png)



![png](output_26_555.png)



![png](output_26_556.png)


    Step 6650: Generator loss: 89.2041259765625, critic loss: -12.506926847219471



![png](output_26_558.png)



![png](output_26_559.png)



![png](output_26_560.png)


    Step 6700: Generator loss: 83.99822830200195, critic loss: -15.094380576133728



![png](output_26_562.png)



![png](output_26_563.png)



![png](output_26_564.png)


    Step 6750: Generator loss: 85.8144744873047, critic loss: -2.740171830177307



![png](output_26_566.png)



![png](output_26_567.png)



![png](output_26_568.png)


    Step 6800: Generator loss: 84.0900946044922, critic loss: -5.654044689059258



![png](output_26_570.png)



![png](output_26_571.png)



![png](output_26_572.png)


    Step 6850: Generator loss: 82.74181716918946, critic loss: -17.15608220529556



![png](output_26_574.png)



![png](output_26_575.png)



![png](output_26_576.png)


    Step 6900: Generator loss: 82.73575180053712, critic loss: -14.952022145271304



![png](output_26_578.png)



![png](output_26_579.png)



![png](output_26_580.png)


    Step 6950: Generator loss: 87.376572265625, critic loss: -14.130860904693602



![png](output_26_582.png)



![png](output_26_583.png)



![png](output_26_584.png)


    Step 7000: Generator loss: 80.39826049804688, critic loss: -21.454723683595653



![png](output_26_586.png)



![png](output_26_587.png)



![png](output_26_588.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 7050: Generator loss: 85.81038146972656, critic loss: -13.878658952236174



![png](output_26_592.png)



![png](output_26_593.png)



![png](output_26_594.png)


    Step 7100: Generator loss: 81.9545980834961, critic loss: -18.298397474288937



![png](output_26_596.png)



![png](output_26_597.png)



![png](output_26_598.png)


    Step 7150: Generator loss: 89.20391998291015, critic loss: -7.00580557346344



![png](output_26_600.png)



![png](output_26_601.png)



![png](output_26_602.png)


    Step 7200: Generator loss: 88.05260223388672, critic loss: -9.111673758506775



![png](output_26_604.png)



![png](output_26_605.png)



![png](output_26_606.png)


    Step 7250: Generator loss: 90.50775436401368, critic loss: -12.744439420700074



![png](output_26_608.png)



![png](output_26_609.png)



![png](output_26_610.png)


    Step 7300: Generator loss: 87.51148696899413, critic loss: -16.15311605501175



![png](output_26_612.png)



![png](output_26_613.png)



![png](output_26_614.png)


    Step 7350: Generator loss: 84.04835693359375, critic loss: -17.066434749126433



![png](output_26_616.png)



![png](output_26_617.png)



![png](output_26_618.png)


    Step 7400: Generator loss: 84.0791812133789, critic loss: -18.158744814395902



![png](output_26_620.png)



![png](output_26_621.png)



![png](output_26_622.png)


    Step 7450: Generator loss: 86.21021224975586, critic loss: -15.110127362012863



![png](output_26_624.png)



![png](output_26_625.png)



![png](output_26_626.png)


    Step 7500: Generator loss: 88.12759887695313, critic loss: -16.239437568187714



![png](output_26_628.png)



![png](output_26_629.png)



![png](output_26_630.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 7550: Generator loss: 88.66850402832031, critic loss: -10.484792006969451



![png](output_26_634.png)



![png](output_26_635.png)



![png](output_26_636.png)


    Step 7600: Generator loss: 86.94520812988281, critic loss: -18.572328203201298



![png](output_26_638.png)



![png](output_26_639.png)



![png](output_26_640.png)


    Step 7650: Generator loss: 89.63778228759766, critic loss: -18.47172679758072



![png](output_26_642.png)



![png](output_26_643.png)



![png](output_26_644.png)


    Step 7700: Generator loss: 97.44354064941406, critic loss: -20.024252505302435



![png](output_26_646.png)



![png](output_26_647.png)



![png](output_26_648.png)


    Step 7750: Generator loss: 84.24808090209962, critic loss: -10.397000555038455



![png](output_26_650.png)



![png](output_26_651.png)



![png](output_26_652.png)


    Step 7800: Generator loss: 84.19105697631836, critic loss: -9.129007452964782



![png](output_26_654.png)



![png](output_26_655.png)



![png](output_26_656.png)


    Step 7850: Generator loss: 90.18692993164062, critic loss: -10.78334185886383



![png](output_26_658.png)



![png](output_26_659.png)



![png](output_26_660.png)


    Step 7900: Generator loss: 87.95388046264648, critic loss: -14.32978438973427



![png](output_26_662.png)



![png](output_26_663.png)



![png](output_26_664.png)


    Step 7950: Generator loss: 88.98179306030273, critic loss: -12.640806052207948



![png](output_26_666.png)



![png](output_26_667.png)



![png](output_26_668.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 8000: Generator loss: 84.84067291259765, critic loss: -16.862181128025057



![png](output_26_672.png)



![png](output_26_673.png)



![png](output_26_674.png)


    Step 8050: Generator loss: 86.96178756713867, critic loss: -13.361723540544508



![png](output_26_676.png)



![png](output_26_677.png)



![png](output_26_678.png)


    Step 8100: Generator loss: 82.57325866699219, critic loss: -18.359965005159378



![png](output_26_680.png)



![png](output_26_681.png)



![png](output_26_682.png)


    Step 8150: Generator loss: 88.49420257568359, critic loss: -15.538886476993557



![png](output_26_684.png)



![png](output_26_685.png)



![png](output_26_686.png)


    Step 8200: Generator loss: 92.5031381225586, critic loss: 2.162937791109086



![png](output_26_688.png)



![png](output_26_689.png)



![png](output_26_690.png)


    Step 8250: Generator loss: 91.00879974365235, critic loss: -3.9574946012496937



![png](output_26_692.png)



![png](output_26_693.png)



![png](output_26_694.png)


    Step 8300: Generator loss: 81.383818359375, critic loss: -14.418639464855197



![png](output_26_696.png)



![png](output_26_697.png)



![png](output_26_698.png)


    Step 8350: Generator loss: 84.02410125732422, critic loss: -17.72073389720917



![png](output_26_700.png)



![png](output_26_701.png)



![png](output_26_702.png)


    Step 8400: Generator loss: 85.80985610961915, critic loss: -16.22680687355995



![png](output_26_704.png)



![png](output_26_705.png)



![png](output_26_706.png)



```python

```
