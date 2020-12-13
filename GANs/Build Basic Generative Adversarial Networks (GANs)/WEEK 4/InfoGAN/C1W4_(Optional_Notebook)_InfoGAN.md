# InfoGAN
*Please note that this is an optional notebook meant to introduce more advanced concepts. If you’re up for a challenge, take a look and don’t worry if you can’t follow everything. There is no code to implement—only some cool code for you to learn and run!*

### Goals

In this notebook, you're going to learn about InfoGAN in order to generate disentangled outputs, based on the paper, [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) by Chen et. al. While there are many approaches to disentanglement, this is one of the more widely used and better known. 

InfoGAN can be understood like this: you want to separate your model into two parts: $z$, corresponding to truly random noise, and $c$ corresponding to the "latent code." The latent code $c$ which can be thought of as a "hidden" condition in a conditional generator, and you'd like it to have an interpretable meaning. 

Now, you'll likely immediately wonder, how do they get $c$, which is just some random set of numbers, to be more interpretable than any dimension in a typical GAN? The answer is "mutual information": essentially, you would like each dimension of the latent code to be as obvious a function as possible of the generated images. Read on for a more thorough theoretical and practical treatment.

### Formally: Variational Lower Bound
The [information entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) ${H} (X)=-\sum _{i=1}^{n}{P(x_{i})\log P (x_{i})}$
can be understood to the amount of "information" in the distribution $X$. For example, the information entropy of $n$ fair coins is $n$ bits. You've also seen a similar equation before: the cross-entropy loss. Moreover, mutual information $I(X;Y) = H(X) - H(X\vert Y)$, which the authors of InfoGAN describe as (intuitively) the "reduction of uncertainty in $X$ when $Y$ is observed." 

In InfoGAN, you'd like to maximize $I(c; G(z, c))$, the mutual information between the latent code $c$ and the generated images $G(z, c)$.  Since it's difficult to know $P(c | G(z, c))$, you add a second output to the discriminator to predict $P(c | G(z, c))$. 

Let $\Delta = D_{KL}(P(\cdot|x) \Vert Q(\cdot|x))$, the [Kullback-Leibler_divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the true and approximate distribution. Then, based on Equation 4 in the paper, the mutual information has the following lower bound: 
$$\begin{split}
I(c; G(z, c)) & = H(c) - H(c|G(z, c)) \\
& = {\mathbb{E}}_{x \sim G(z, c)} [ {\mathbb{E}}_{c' \sim P(c, x)} \log P(c' | x) ] + H(c) \textit{ (by definition of H)}\\
& = {\mathbb{E}}_{x \sim G(z, c)} [\Delta + {\mathbb{E}}_{c' \sim P(c, x)} \log Q(c' | x) ] + H(c) \textit{ (approximation error)}\\
& \geq {\mathbb{E}}_{x \sim G(z, c)} [{\mathbb{E}}_{c' \sim P(c, x)} \log Q(c' | x) ] + H(c) \textit{ (KL divergence is non-negative)}\\
\end{split}
$$

For a given latent code distribution, $H(c)$ is fixed, so the following makes a good loss:

$${\mathbb{E}}_{x \sim G(z, c)} [{\mathbb{E}}_{c' \sim P(c, x)} \log Q(c' | x) ]$$

Which is the mean cross entropy loss of the approximation over the generator's images. 

### Updating the Minimax Game

A vanilla generator and discriminator follow a minimax game: $\displaystyle \min_{G} \max_{D} V(D, G) = \mathbb{E}(\log D(x)) + \mathbb{E}(\log (1 - D(G(z))))$.

To encourage mutual information, this game is updated for $Q$ to maximize mutual information: $\displaystyle \min_{G, Q} \max_{D} V(D, G) - \lambda I(c; G(z, c))$

## Implementing InfoGAN

For this notebook, you'll be using the MNIST dataset again. 

You will begin by importing the necessary libraries and building the generator and discriminator. The generator will be the same as before, but the discriminator will be modified with more dimensions in its output.

#### Packages and Visualization


```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for our testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
```

#### Generator and Noise


```python
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
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
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, input_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)
```

#### InfoGAN Discriminator

You update the final layer to predict a distribution for $c$ from $x$, alongside the traditional discriminator output. Since you're assuming a normal prior in this assignment, you output a mean and a log-variance prediction.


```python
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
      im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel is your default)
      hidden_dim: the inner dimension, a scalar
      c_dim: the number of latent code dimensions - 
    '''
    def __init__(self, im_chan=1, hidden_dim=64, c_dim=10):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
        )
        self.d_layer = self.make_disc_block(hidden_dim * 2, 1, final_layer=True)
        self.q_layer = nn.Sequential(
            self.make_disc_block(hidden_dim * 2, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 2 * c_dim, kernel_size=1, final_layer=True)
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN; 
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
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        intermediate_pred = self.disc(image)
        disc_pred = self.d_layer(intermediate_pred)
        q_pred = self.q_layer(intermediate_pred)
        return disc_pred.view(len(disc_pred), -1), q_pred.view(len(q_pred), -1)
```

## Helper Functions

You can include some helper functions for conditional GANs:


```python
def combine_vectors(x, y):
    '''
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector. 
        This will be the noise vector of shape (n_samples, z_dim).
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector 
        with the shape (n_samples, n_classes).
    '''
    combined = torch.cat([x.float(), y.float()], 1)
    return combined
```

## Training

Let's include the same parameters from previous assignments, as well as a new `c_dim` dimension for the dimensionality of the InfoGAN latent code, a `c_criterion`, and its corresponding constant, `c_lambda`:

  *   mnist_shape: the number of pixels in each MNIST image, which has dimensions 28 x 28 and one channel (because it's black-and-white) so 1 x 28 x 28
  *   adv_criterion: the vanilla GAN loss function
  *   c_criterion: the additional mutual information term
  *   c_lambda: the weight on the c_criterion
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   c_dim: the dimension of the InfoGAN latent code
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   device: the device type



```python
from torch.distributions.normal import Normal
adv_criterion = nn.BCEWithLogitsLoss()
c_criterion = lambda c_true, mean, logvar: Normal(mean, logvar.exp()).log_prob(c_true).mean()
c_lambda = 0.1
mnist_shape = (1, 28, 28)
n_epochs = 80
z_dim = 64
c_dim = 2
display_step = 500
batch_size = 128
# InfoGAN uses two different learning rates for the models
d_lr = 2e-4
g_lr = 1e-3
device = 'cuda'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)
```

You initialize your networks as usual - notice that there is no separate $Q$ network. There are a few "design" choices worth noting here: 
1. There are many possible choices for the distribution over the latent code. You use a Gaussian prior here, but a categorical (discrete) prior is also possible, and in fact it's possible to use them together. In this case, it's also possible to use different weights $\lambda$ on both prior distributions. 
2. You can calculate the mutual information explicitly, including $H(c)$ which you treat as constant here. You don't do that here since you're not comparing the mutual information of different parameterizations of the latent code.
3. There are multiple ways to handle the $Q$ network - this code follows the original paper by treating it as part of the discriminator, sharing most weights, but it is also possible to simply initialize another network.


```python
gen = Generator(input_dim=z_dim + c_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=g_lr)
disc = Discriminator(im_chan=mnist_shape[0], c_dim=c_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=d_lr)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
```

Now let's get to training the networks:


```python
cur_step = 0
generator_losses = []
discriminator_losses = []

for epoch in range(n_epochs):
    # Dataloader returns the batches and the labels
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        # Flatten the batch of real images from the dataset
        real = real.to(device)

        c_labels = get_noise(cur_batch_size, c_dim, device=device)    
        ### Update discriminator ###
        # Zero out the discriminator gradients
        disc_opt.zero_grad()
        # Get noise corresponding to the current batch_size 
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        # Combine the noise vectors and the one-hot labels for the generator
        noise_and_labels = combine_vectors(fake_noise, c_labels)
        # Generate the conditioned fake images
        fake = gen(noise_and_labels)
        
        # Get the discriminator's predictions
        disc_fake_pred, disc_q_pred = disc(fake.detach())
        disc_q_mean = disc_q_pred[:, :c_dim]
        disc_q_logvar = disc_q_pred[:, c_dim:]
        mutual_information = c_criterion(c_labels, disc_q_mean, disc_q_logvar)
        disc_real_pred, _ = disc(real)
        disc_fake_loss = adv_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = adv_criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2 - c_lambda * mutual_information
        disc_loss.backward(retain_graph=True)
        disc_opt.step() 

        # Keep track of the average discriminator loss
        discriminator_losses += [disc_loss.item()]

        ### Update generator ###
        # Zero out the generator gradients
        gen_opt.zero_grad()

        disc_fake_pred, disc_q_pred = disc(fake)
        disc_q_mean = disc_q_pred[:, :c_dim]
        disc_q_logvar = disc_q_pred[:, c_dim:]
        mutual_information = c_criterion(c_labels, disc_q_mean, disc_q_logvar)
        gen_loss = adv_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - c_lambda * mutual_information
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]

        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Discriminator Loss"
            )
            plt.legend()
            plt.show()
        cur_step += 1
```


    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 1, step 500: Generator loss: 0.8030008111000061, discriminator loss: 0.7182435142993927



![png](output_18_4.png)



![png](output_18_5.png)



![png](output_18_6.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 2, step 1000: Generator loss: 1.503768704533577, discriminator loss: 0.32358910217136144



![png](output_18_10.png)



![png](output_18_11.png)



![png](output_18_12.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 3, step 1500: Generator loss: 2.757066939353943, discriminator loss: 0.05517909451574087



![png](output_18_16.png)



![png](output_18_17.png)



![png](output_18_18.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 4, step 2000: Generator loss: 3.3841658539772035, discriminator loss: 0.019859904855489732



![png](output_18_22.png)



![png](output_18_23.png)



![png](output_18_24.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 5, step 2500: Generator loss: 3.5032820949554444, discriminator loss: -0.0012047540117055178



![png](output_18_28.png)



![png](output_18_29.png)



![png](output_18_30.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 6, step 3000: Generator loss: 3.4627780723571777, discriminator loss: 0.09260665814206004



![png](output_18_34.png)



![png](output_18_35.png)



![png](output_18_36.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 7, step 3500: Generator loss: 2.4104007251262667, discriminator loss: 0.3190444306284189



![png](output_18_40.png)



![png](output_18_41.png)



![png](output_18_42.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 8, step 4000: Generator loss: 2.0138914420604705, discriminator loss: 0.34644281095266344



![png](output_18_46.png)



![png](output_18_47.png)



![png](output_18_48.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 9, step 4500: Generator loss: 2.1300189616680147, discriminator loss: 0.32848581594228743



![png](output_18_52.png)



![png](output_18_53.png)



![png](output_18_54.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 10, step 5000: Generator loss: 2.1481298537254334, discriminator loss: 0.3186153033673763



![png](output_18_58.png)



![png](output_18_59.png)



![png](output_18_60.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 11, step 5500: Generator loss: 2.2064590510129927, discriminator loss: 0.3131762868911028



![png](output_18_64.png)



![png](output_18_65.png)



![png](output_18_66.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 12, step 6000: Generator loss: 1.9004919929504394, discriminator loss: 0.33119745552539825



![png](output_18_70.png)



![png](output_18_71.png)



![png](output_18_72.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 13, step 6500: Generator loss: 1.8820792179107666, discriminator loss: 0.3579175540804863



![png](output_18_76.png)



![png](output_18_77.png)



![png](output_18_78.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 14, step 7000: Generator loss: 1.816380264043808, discriminator loss: 0.41256577062606814



![png](output_18_82.png)



![png](output_18_83.png)



![png](output_18_84.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 15, step 7500: Generator loss: 1.689778028845787, discriminator loss: 0.4031879070401192



![png](output_18_88.png)



![png](output_18_89.png)



![png](output_18_90.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 17, step 8000: Generator loss: 1.728199395775795, discriminator loss: 0.4328122897297144



![png](output_18_96.png)



![png](output_18_97.png)



![png](output_18_98.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 18, step 8500: Generator loss: 1.6771549911499024, discriminator loss: 0.42397291675209997



![png](output_18_102.png)



![png](output_18_103.png)



![png](output_18_104.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 19, step 9000: Generator loss: 1.6297683087587356, discriminator loss: 0.4425521269440651



![png](output_18_108.png)



![png](output_18_109.png)



![png](output_18_110.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 20, step 9500: Generator loss: 1.5486765871047974, discriminator loss: 0.4822590392827988



![png](output_18_114.png)



![png](output_18_115.png)



![png](output_18_116.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 21, step 10000: Generator loss: 1.4254574308395385, discriminator loss: 0.49669707411527636



![png](output_18_120.png)



![png](output_18_121.png)



![png](output_18_122.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 22, step 10500: Generator loss: 1.4502755423784257, discriminator loss: 0.4937284078896046



![png](output_18_126.png)



![png](output_18_127.png)



![png](output_18_128.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 23, step 11000: Generator loss: 1.4035085102319718, discriminator loss: 0.5058651449680328



![png](output_18_132.png)



![png](output_18_133.png)



![png](output_18_134.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 24, step 11500: Generator loss: 1.375338632583618, discriminator loss: 0.5110890194773674



![png](output_18_138.png)



![png](output_18_139.png)



![png](output_18_140.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 25, step 12000: Generator loss: 1.2952413198947907, discriminator loss: 0.5286408168673515



![png](output_18_144.png)



![png](output_18_145.png)



![png](output_18_146.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 26, step 12500: Generator loss: 1.2801015807390212, discriminator loss: 0.5326109395623208



![png](output_18_150.png)



![png](output_18_151.png)



![png](output_18_152.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 27, step 13000: Generator loss: 1.3084286637306213, discriminator loss: 0.5372092262506485



![png](output_18_156.png)



![png](output_18_157.png)



![png](output_18_158.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 28, step 13500: Generator loss: 1.2267155009508133, discriminator loss: 0.5434517996311188



![png](output_18_162.png)



![png](output_18_163.png)



![png](output_18_164.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 29, step 14000: Generator loss: 1.2536467809677123, discriminator loss: 0.5497834231853486



![png](output_18_168.png)



![png](output_18_169.png)



![png](output_18_170.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 30, step 14500: Generator loss: 1.2784405480623244, discriminator loss: 0.5489084910154343



![png](output_18_174.png)



![png](output_18_175.png)



![png](output_18_176.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 31, step 15000: Generator loss: 1.2144795101881027, discriminator loss: 0.5507858278155326



![png](output_18_180.png)



![png](output_18_181.png)



![png](output_18_182.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 33, step 15500: Generator loss: 1.3069980521202087, discriminator loss: 0.5434530960917473



![png](output_18_188.png)



![png](output_18_189.png)



![png](output_18_190.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 34, step 16000: Generator loss: 1.256813486456871, discriminator loss: 0.5460377159714699



![png](output_18_194.png)



![png](output_18_195.png)



![png](output_18_196.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 35, step 16500: Generator loss: 1.173333095908165, discriminator loss: 0.5543401240110397



![png](output_18_200.png)



![png](output_18_201.png)



![png](output_18_202.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 36, step 17000: Generator loss: 1.2031154154539108, discriminator loss: 0.544552142560482



![png](output_18_206.png)



![png](output_18_207.png)



![png](output_18_208.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 37, step 17500: Generator loss: 1.2537568380832673, discriminator loss: 0.5433059730529786



![png](output_18_212.png)



![png](output_18_213.png)



![png](output_18_214.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 38, step 18000: Generator loss: 1.2122063930034637, discriminator loss: 0.5403282681107521



![png](output_18_218.png)



![png](output_18_219.png)



![png](output_18_220.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 39, step 18500: Generator loss: 1.174001025080681, discriminator loss: 0.5360180272459983



![png](output_18_224.png)



![png](output_18_225.png)



![png](output_18_226.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 40, step 19000: Generator loss: 1.171091105222702, discriminator loss: 0.5357524607777595



![png](output_18_230.png)



![png](output_18_231.png)



![png](output_18_232.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 41, step 19500: Generator loss: 1.1854595437049866, discriminator loss: 0.5335679452419281



![png](output_18_236.png)



![png](output_18_237.png)



![png](output_18_238.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 42, step 20000: Generator loss: 1.1901087317466736, discriminator loss: 0.5384431281089783



![png](output_18_242.png)



![png](output_18_243.png)



![png](output_18_244.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 43, step 20500: Generator loss: 1.1551499643325807, discriminator loss: 0.5338953946232796



![png](output_18_248.png)



![png](output_18_249.png)



![png](output_18_250.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 44, step 21000: Generator loss: 1.1737536680698395, discriminator loss: 0.5273768262267112



![png](output_18_254.png)



![png](output_18_255.png)



![png](output_18_256.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 45, step 21500: Generator loss: 1.1972835119962693, discriminator loss: 0.5281104121208191



![png](output_18_260.png)



![png](output_18_261.png)



![png](output_18_262.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 46, step 22000: Generator loss: 1.1820068897008895, discriminator loss: 0.5253298255801201



![png](output_18_266.png)



![png](output_18_267.png)



![png](output_18_268.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 47, step 22500: Generator loss: 1.1569786469936372, discriminator loss: 0.520203008890152



![png](output_18_272.png)



![png](output_18_273.png)



![png](output_18_274.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 49, step 23000: Generator loss: 1.2006243673563004, discriminator loss: 0.5256361476182938



![png](output_18_280.png)



![png](output_18_281.png)



![png](output_18_282.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 50, step 23500: Generator loss: 1.174626919388771, discriminator loss: 0.5252262147068978



![png](output_18_286.png)



![png](output_18_287.png)



![png](output_18_288.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 51, step 24000: Generator loss: 1.1920923249721527, discriminator loss: 0.5303788462281227



![png](output_18_292.png)



![png](output_18_293.png)



![png](output_18_294.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 52, step 24500: Generator loss: 1.1695511214733123, discriminator loss: 0.523569656074047



![png](output_18_298.png)



![png](output_18_299.png)



![png](output_18_300.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 53, step 25000: Generator loss: 1.1820153337717056, discriminator loss: 0.5273034728169441



![png](output_18_304.png)



![png](output_18_305.png)



![png](output_18_306.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 54, step 25500: Generator loss: 1.152515373468399, discriminator loss: 0.5374031292200089



![png](output_18_310.png)



![png](output_18_311.png)



![png](output_18_312.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 55, step 26000: Generator loss: 1.1893082225322724, discriminator loss: 0.5337966603040695



![png](output_18_316.png)



![png](output_18_317.png)



![png](output_18_318.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 56, step 26500: Generator loss: 1.1567092360258102, discriminator loss: 0.5423419881463051



![png](output_18_322.png)



![png](output_18_323.png)



![png](output_18_324.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 57, step 27000: Generator loss: 1.1414481619596482, discriminator loss: 0.5346725415587426



![png](output_18_328.png)



![png](output_18_329.png)



![png](output_18_330.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 58, step 27500: Generator loss: 1.1373897230625152, discriminator loss: 0.5428345699310303



![png](output_18_334.png)



![png](output_18_335.png)



![png](output_18_336.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 59, step 28000: Generator loss: 1.1630004214048386, discriminator loss: 0.5344629813432693



![png](output_18_340.png)



![png](output_18_341.png)



![png](output_18_342.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 60, step 28500: Generator loss: 1.1463847043514253, discriminator loss: 0.535941004395485



![png](output_18_346.png)



![png](output_18_347.png)



![png](output_18_348.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 61, step 29000: Generator loss: 1.1631728945970534, discriminator loss: 0.5328843494653702



![png](output_18_352.png)



![png](output_18_353.png)



![png](output_18_354.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 62, step 29500: Generator loss: 1.138320455789566, discriminator loss: 0.5363343282341957



![png](output_18_358.png)



![png](output_18_359.png)



![png](output_18_360.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 63, step 30000: Generator loss: 1.1587439556121826, discriminator loss: 0.5432113043665886



![png](output_18_364.png)



![png](output_18_365.png)



![png](output_18_366.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 65, step 30500: Generator loss: 1.1348646243810654, discriminator loss: 0.5346470260024071



![png](output_18_372.png)



![png](output_18_373.png)



![png](output_18_374.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 66, step 31000: Generator loss: 1.176431158542633, discriminator loss: 0.5336210044622421



![png](output_18_378.png)



![png](output_18_379.png)



![png](output_18_380.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 67, step 31500: Generator loss: 1.1499754862785339, discriminator loss: 0.5418602007031441



![png](output_18_384.png)



![png](output_18_385.png)



![png](output_18_386.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 68, step 32000: Generator loss: 1.172464220404625, discriminator loss: 0.5323853597044945



![png](output_18_390.png)



![png](output_18_391.png)



![png](output_18_392.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 69, step 32500: Generator loss: 1.1792231738567351, discriminator loss: 0.5369196996092797



![png](output_18_396.png)



![png](output_18_397.png)



![png](output_18_398.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 70, step 33000: Generator loss: 1.1643480468988419, discriminator loss: 0.5396220123767853



![png](output_18_402.png)



![png](output_18_403.png)



![png](output_18_404.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 71, step 33500: Generator loss: 1.1179705241918563, discriminator loss: 0.5383707079291343



![png](output_18_408.png)



![png](output_18_409.png)



![png](output_18_410.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 72, step 34000: Generator loss: 1.127301449775696, discriminator loss: 0.5301177815198899



![png](output_18_414.png)



![png](output_18_415.png)



![png](output_18_416.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 73, step 34500: Generator loss: 1.1236147677898407, discriminator loss: 0.5355471240282059



![png](output_18_420.png)



![png](output_18_421.png)



![png](output_18_422.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 74, step 35000: Generator loss: 1.1800841537714004, discriminator loss: 0.5332254801392555



![png](output_18_426.png)



![png](output_18_427.png)



![png](output_18_428.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 75, step 35500: Generator loss: 1.1514811514616012, discriminator loss: 0.530103328704834



![png](output_18_432.png)



![png](output_18_433.png)



![png](output_18_434.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 76, step 36000: Generator loss: 1.1482917863130568, discriminator loss: 0.5347169573903083



![png](output_18_438.png)



![png](output_18_439.png)



![png](output_18_440.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 77, step 36500: Generator loss: 1.1055165784358978, discriminator loss: 0.5445700452923775



![png](output_18_444.png)



![png](output_18_445.png)



![png](output_18_446.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 78, step 37000: Generator loss: 1.1278807399272919, discriminator loss: 0.537358128964901



![png](output_18_450.png)



![png](output_18_451.png)



![png](output_18_452.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 79, step 37500: Generator loss: 1.1290170905590058, discriminator loss: 0.5314615268111229



![png](output_18_456.png)



![png](output_18_457.png)



![png](output_18_458.png)


    


## Exploration
You can do a bit of exploration now!


```python
# Before you explore, you should put the generator
# in eval mode, both in general and so that batch norm
# doesn't cause you issues and is using its eval statistics
gen = gen.eval()
```

#### Changing the Latent Code Vector
You can generate some numbers with your new model! You can add interpolation as well to make it more interesting.

So starting from a image, you will produce intermediate images that look more and more like the ending image until you get to the final image. Your're basically morphing one image into another. You can choose what these two images will be using your conditional GAN.


```python
import math

### Change me! ###
n_interpolation = 9 # Choose the interpolation: how many intermediate images you want + 2 (for the start and end image)

def interpolate_class(n_view=5):
    interpolation_noise = get_noise(n_view, z_dim, device=device).repeat(n_interpolation, 1)
    first_label = get_noise(1, c_dim).repeat(n_view, 1)[None, :]
    second_label = first_label.clone()
    first_label[:, :, 0] =  -2
    second_label[:, :, 0] =  2
    

    # Calculate the interpolation vector between the two labels
    percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None, None]
    interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label
    interpolation_labels = interpolation_labels.view(-1, c_dim)

    # Combine the noise and the labels
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation * n_view, nrow=n_view, show=False)

plt.figure(figsize=(8, 8))
interpolate_class()
_ = plt.axis('off')

```


![png](output_22_0.png)


You can also visualize the impact of pairwise changes of the latent code for a given noise vector.


```python
import math

### Change me! ###
n_interpolation = 8 # Choose the interpolation: how many intermediate images you want + 2 (for the start and end image)

def interpolate_class():
    interpolation_noise = get_noise(1, z_dim, device=device).repeat(n_interpolation * n_interpolation, 1)
    first_label = get_noise(1, c_dim).repeat(n_interpolation * n_interpolation, 1)
    
    # Calculate the interpolation vector between the two labels
    first_label = torch.linspace(-2, 2, n_interpolation).repeat(n_interpolation)
    second_label = torch.linspace(-2, 2, n_interpolation).repeat_interleave(n_interpolation)
    interpolation_labels = torch.stack([first_label, second_label], dim=1) 

    # Combine the noise and the labels
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation * n_interpolation, nrow=n_interpolation, show=False)

plt.figure(figsize=(8, 8))
interpolate_class()
_ = plt.axis('off')

```


![png](output_24_0.png)

