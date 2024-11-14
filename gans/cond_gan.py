import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import sys
from discrimator import *
from generator import *


torch.manual_seed(0) # Set for our testing purposes, please do not change!
mnist_shape = (1, 28, 28)
n_classes = 10
criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
z_dim = 64
display_step = 10
batch_size = 128
lr = 0.0002
device = 'cpu'
num_workers = 16

torch.set_num_threads(num_workers)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
    )






def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, num_classes=n_classes)

def combine_vectors(x, y):
    combined = torch.cat((x.float(),y.float()),dim=1)
    return combined

def show_tensor_images(image_tensor, num_images=100, size=(1, 28, 28), nrow=10, show=False, save_path=None):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save the image grid to a file
    plt.clf()

def get_input_dimensions(z_dim, mnist_shape, n_classes):    
    generator_input_dim = z_dim+n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)
gen = Generator(input_dim=generator_input_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


cur_step = 0
generator_losses = []
discriminator_losses = []
noise_and_labels = False
fake = False

fake_image_and_labels = False
real_image_and_labels = False
disc_fake_pred = False
disc_real_pred = False

for epoch in range(n_epochs):
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)
        one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

        disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        fake = gen(noise_and_labels)
        assert len(fake) == len(real)
        assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
        assert tuple(fake.shape) == (len(real), 1, 28, 28)

        fake_image_and_labels = torch.cat((fake.detach(), image_one_hot_labels.float()), dim=1)
        real_image_and_labels = torch.cat((real.to(device), image_one_hot_labels.float()), dim=1)
        disc_fake_pred = disc(fake_image_and_labels)
        disc_real_pred = disc(real_image_and_labels)
       
        
        assert tuple(fake_image_and_labels.shape) == (len(real), fake.detach().shape[1] + image_one_hot_labels.shape[1], 28 ,28)
        assert tuple(real_image_and_labels.shape) == (len(real), real.shape[1] + image_one_hot_labels.shape[1], 28 ,28)
        assert len(disc_real_pred) == len(real)
        assert torch.any(fake_image_and_labels != real_image_and_labels)
        assert tuple(fake_image_and_labels.shape) == tuple(real_image_and_labels.shape)
        assert tuple(disc_fake_pred.shape) == tuple(disc_real_pred.shape)
        
        
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step() 

        # Keep track of the average discriminator loss
        discriminator_losses += [disc_loss.item()]

        ### Update generator ###
        
        gen_opt.zero_grad()

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        # This will error if you didn't concatenate your labels to your image correctly
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]
        #
        if cur_step == 0:
            print("Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!")
        cur_step += 1
        

    if epoch % display_step == 0 and epoch > 0:
        model_path = f"gans/results/generator_{epoch}.pth"
        torch.save(gen.state_dict(), model_path)
        gen_mean = sum(generator_losses[-display_step:]) / display_step
        disc_mean = sum(discriminator_losses[-display_step:]) / display_step
        print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
        save_path_fake = f"gans/results/fake_{epoch}.png"
        save_path_real = f"gans/results/real_{epoch}.png"
        show_tensor_images(fake, save_path=save_path_fake)
        show_tensor_images(real, save_path=save_path_real)
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
        # plt.show()
        loss_plot_path = f"gans/results/loss_plot_{epoch}.png"
        plt.savefig(loss_plot_path, bbox_inches='tight')
        plt.clf()
    