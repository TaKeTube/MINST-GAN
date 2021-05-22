# Reference: Pytorch's DCGAN Tutorial
# URL: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Modified by Guan Zimu

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Root directory for dataset
dataroot = "data/MNIST"

# Number of workers for dataloader
workers = 8

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
Glr = 0.0002
Dlr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of rows of the result image
nrow = 10

# ==================
# initialize weights
# ==================

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# =========
# Generator
# =========

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# =============
# Discriminator
# =============

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

if __name__=='__main__':

    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # =====================
    # load & visualize data
    # =====================

    # Create the dataset
    trans = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
    # load training data
    training_data = dset.MNIST(root=dataroot, train=True, transform=trans, download=True)
    # load testing data
    testing_data = dset.MNIST(root=dataroot, train=False, transform=trans)
    # combine two data sets
    dataset = training_data + testing_data
    print("Size of datasize : ", len(dataset))

    # Create the dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers
                            # pin_memory=True
                            )

    # show some traning data
    # real_batch = next(iter(dataloader))[0]
    # plt.figure(figsize=(10,10))
    # plt.title("Training Images")
    # plt.axis('off')
    # inputs = vutils.make_grid(real_batch[:nrow**2], nrow=nrow)
    # plt.imshow(inputs.permute(1, 2, 0))

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # =========
    # Generator
    # =========

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # =============
    # Discriminator
    # =============

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # =============================
    # Loss Functions and Optimizers
    # =============================

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(nrow**2, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    # real_label = 0.9
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=Dlr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=Glr, betas=(beta1, 0.999))

    # ========
    # Training
    # ========

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    D_x_list = []
    D_G_z2_list = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        begin_time = time.time()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            # ===========================================================
            # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # ===========================================================
            
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # ==========================================
            # 2. Update Generator: maximize log(D(G(z)))
            # ==========================================
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            end_time = time.time()
            run_time = int(end_time - begin_time)
            if i % 50 == 0:
                print('Epoch: [%d/%d]\tStep: [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tTime: %ds'
                        % (epoch, num_epochs, i, len(dataloader), 
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1, D_G_z2,
                        run_time)
                        )
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            D_x_list.append(D_x)
            D_G_z2_list.append(D_G_z2)
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=nrow))
                
            iters += 1

    # =======
    # Results
    # =======

    # plot loss of each epoch
    plt.figure(figsize=(18,10))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.axhline(y=0, label="0")
    plt.legend()
    # plt.show()

    plt.savefig('./results/loss.png')
    plt.close()

    # plot loss of each epoch
    plt.figure(figsize=(18,10))
    plt.title("D(x) & D(G(z)) During Training")
    plt.plot(D_x_list,label="D(x)")
    plt.plot(D_G_z2_list,label="D(G(z))")
    plt.xlabel("iterations")
    plt.ylabel("Probability")
    plt.axhline(y=0.5, label="0.5")
    plt.legend()
    # plt.show()

    plt.savefig('./results/probability.png')
    plt.close()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(vutils.make_grid(real_batch[0][:nrow**2], nrow=nrow).permute(1, 2, 0))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    fake = vutils.make_grid(img_list[-1], nrow=nrow)
    plt.imshow(fake.permute(1, 2, 0))
    # plt.show()

    plt.savefig('./results/compare.png')
    plt.close()

    # save images
    idx = 0
    plt.figure()
    for i in img_list:
        plt.imshow(np.transpose(i,(1,2,0)), animated=True)
        plt.axis('off')
        plt.savefig('./results/'+str(idx)+'.png')
        plt.close()
        idx += 1

    # # plot traning process animation
    # fig = plt.figure(figsize=(8,8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    # HTML(ani.to_jshtml())

    # save the generator & discriminator
    torch.save(netD, './netD.pkl')
    torch.save(netG, './netG.pkl')
