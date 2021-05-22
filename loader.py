import random
import numpy as np
import os
import gc

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
batch_size = 256

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
num_epochs = 1

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

# number of images
img_num = 100

# load model
model_path = './eval/netG.pkl'
netG = torch.load(model_path)

# get device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# The following code is used for generating and saving fake images for the sake of evaluation
# Be careful for your RAM, do not save fake & real images at the same time

for i in range(img_num):
    # Generate the Fake Images
    noise = torch.randn(nrow**2, nz, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(noise)
    fake = vutils.make_grid(fake, nrow=nrow)

    vutils.save_image(fake.cpu(), './eval/fake/'+str(i)+'.png')


# The following code is used for saving real images for the sake of evaluation
# Be careful for your RAM, do not save fake & real images at the same time

# trans = transforms.Compose([
#                             transforms.Resize(image_size),
#                             transforms.CenterCrop(image_size),
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5,), (0.5,)),
#                             ])
# # load training data
# training_data = dset.MNIST(root=dataroot, train=True, transform=trans, download=True)
# # load testing data
# testing_data = dset.MNIST(root=dataroot, train=False, transform=trans)
# # combine two data sets
# dataset = training_data + testing_data


# dataloader = DataLoader(dataset,
#                         batch_size=1024,
#                         shuffle=True,
#                         num_workers=workers
#                         # pin_memory=True
#                         )

# real_batch = next(iter(dataloader))[0]

# for i in range(img_num):
#     real = vutils.make_grid(real_batch[i*100:(i+1)*100], nrow=10)
#     vutils.save_image(real.cpu(), './eval/real/'+str(i)+'.png')
