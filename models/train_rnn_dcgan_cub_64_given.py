from __future__ import print_function

import argparse
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from matplotlib.pyplot import figure
from torch.autograd import Variable
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel

from models.bcs import BucketManager
from models.dataset import Text2ImageDataset
from models.training_params import params
from models.utils import weights_init, set_seed


class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2,  3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, embed_vector, z):
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.netG(latent_vector)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16

        self.netD_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        self.netD_2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp, embed):
        x_intermediate = self.netD_1(inp)
        x = self.projector(x_intermediate, embed)
        x = self.netD_2(x)

        return x.view(-1, 1).squeeze(1), x_intermediate


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--verbose"])


BUCKET_NAME = "blawml-caseoutcomes"

beta1 = 0.5


def smooth_label(tensor, offset):
    return tensor + offset


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='BERT CRF example')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank when doing multi-process training, set to -1 if not')
    parser.add_argument('--init-retries', type=int, default=10)
    parser.add_argument('--model_name', type=str)

    return parser.parse_args()


def main():
    print("Starting the main process....", flush=True)
    figure(figsize=(8, 8), dpi=300)
    set_seed(42)
    args = parse_args()

    torch.autograd.set_detect_anomaly(True)

    print("Before environment read....", flush=True)
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    print("Local Rank: {}, Rank: {}".format(args.local_rank, rank), flush=True)

    bucket_manager = BucketManager(BUCKET_NAME)

    print("Initializing torch distributed....", flush=True)
    for _ in range(args.init_retries):
        try:
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=rank
            )
            print("Initialized torch distributed", flush=True)
            break
        except ValueError:
            print("Retrying connecting to master", flush=True)
            time.sleep(1)

    print("Downloading datasets....", flush=True)
    os.mkdir('images')
    bucket_manager.download_object("cub.pkl", "cub.pkl")
    print("Finished downloading dataset", flush=True)

    dataset = Text2ImageDataset(
        datasetFile="cub.pkl"
    )
    print("Finished preparing dataset object", flush=True)

    print('Number of samples: ', len(dataset), flush=True)
    img, captions, cap_lens, fake_img_sample, fake_captions_sample, text = dataset[3]

    print("Image Size: ", img.size(), flush=True)
    print(captions.shape, flush=True)

    print("Preparing dataloader", flush=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        sampler=sampler,
        num_workers=world_size,
        pin_memory=True,
        shuffle=False
    )

    print("Finished preparing data loader".format(rank), flush=True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")

    print("Finished setting device".format(rank), flush=True)

    # Create the generator
    netG = Generator()

    # Create the disciminator
    netD = Discriminator()

    if (device.type == 'cuda') and (params['ngpu'] > 1) and rank >= 0:
        netG = netG.to(device)
        netG = DistributedDataParallel(
            netG,
            device_ids=[0]
        )

        netD.to(device)
        netD = DistributedDataParallel(
            netD,
            device_ids=[0],
            broadcast_buffers=False
        )
    else:
        netG = DistributedDataParallel(netG)
        netD = DistributedDataParallel(netD)

    print("Finished loading netG DDP".format(rank), flush=True)
    netG.apply(weights_init)

    # Print the model
    print(netG)

    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    fixed_noise = torch.randn(64, 100, 1, 1).to(device)
    fixed_images, fixed_captions, fixed_cap_lens, _, _, _ = iter(dataloader).next()

    print(fixed_captions.shape, fixed_cap_lens.shape, flush=True)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(beta1, 0.999))

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0

    print("Inside barrier {}".format(rank), flush=True)
    torch.distributed.barrier()
    print("Exited barrier {}".format(rank), flush=True)

    print("Starting Training Loop...", flush=True)
    # For each epoch
    for epoch in range(params['num_epochs']):
        start = time.time()

        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader, 0):

            right_images = data[0].to(device)
            right_embed = data[1].to(device)
            wrong_images = data[3].to(device)

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()
            wrong_images = Variable(wrong_images.float()).cuda()

            real_labels = torch.ones(right_images.size(0))
            fake_labels = torch.zeros(right_images.size(0))

            # ======== One sided label smoothing ==========
            # Helps preventing the discriminator from overpowering the
            # generator adding penalty when the discriminator is too confident
            # =============================================
            smoothed_real_labels = torch.FloatTensor(smooth_label(real_labels.numpy(), -0.1))

            real_labels = Variable(real_labels).cuda()
            smoothed_real_labels = Variable(smoothed_real_labels).cuda()
            fake_labels = Variable(fake_labels).cuda()

            # Train the discriminator
            netD.zero_grad()
            outputs, activation_real = netD(right_images, right_embed)
            real_loss = criterion(outputs, smoothed_real_labels)
            real_score = outputs

            outputs, _ = netD(wrong_images, right_embed)
            wrong_loss = criterion(outputs, fake_labels)
            wrong_score = outputs

            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = netG(right_embed, noise)
            outputs, _ = netD(fake_images, right_embed)
            fake_loss = criterion(outputs, fake_labels)
            fake_score = outputs

            d_loss = real_loss + fake_loss
            d_loss = d_loss + wrong_loss

            d_loss.backward()
            optimizerD.step()

            # Train the generator
            netG.zero_grad()
            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = netG(right_embed, noise)
            outputs, activation_fake = netD(fake_images, right_embed)
            _, activation_real = netD(right_images, right_embed)

            activation_fake = torch.mean(activation_fake, 0)
            activation_real = torch.mean(activation_real, 0)

            # ======= Generator Loss function============
            # This is a customized loss function, the first term is the regular cross entropy loss
            # The second term is feature matching loss, this measure the distance between the real and generated
            # images statistics by comparing intermediate layers activations
            # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
            # because it links the embedding feature vector directly to certain pixel values.
            # ===========================================
            g_loss = criterion(outputs, real_labels) \
                     + 100 * l2_loss(activation_fake, activation_real.detach()) \
                     + 50 * l1_loss(fake_images, right_images)

            g_loss.backward()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
                    epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
                    fake_score.data.cpu().mean()), flush=True)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == params['num_epochs'] - 1) and (i == len(dataloader) - 1)):
                if rank == 0:
                    plt.imshow(
                        np.moveaxis(vutils.make_grid(right_images[:64], padding=2, normalize=True).detach().cpu().numpy(),
                                    0, -1),
                        interpolation='nearest'
                    )

                    plt.savefig('images/real_{}_{}.png'.format(i, epoch))
                    bucket_manager.upload_file(
                        'text-captions/{}/images/real_{}_{}.png'.format(args.model_name, i, epoch),
                        'images/real_{}_{}.png'.format(i, epoch)
                    )

                with torch.no_grad():
                    fake = netG(fixed_captions, fixed_noise)

                if rank == 0:
                    plt.imshow(
                        np.moveaxis(vutils.make_grid(fake, padding=2, normalize=True).detach().cpu().numpy(), 0, -1),
                        interpolation='nearest'
                    )
                    plt.savefig('images/{}_{}.png'.format(i, epoch))

                    bucket_manager.upload_file(
                        'text-captions/{}/images/{}_{}.png'.format(args.model_name, i, epoch),
                        'images/{}_{}.png'.format(i, epoch)
                    )

            # print("Finished iteration {}".format(i), flush=True)

        if rank == 0:
            torch.save(netG.module.state_dict(), "netG_{}.model".format(epoch))
            torch.save(netD.module.state_dict(), "netD_{}.model".format(epoch))

            bucket_manager.upload_file(
                'text-captions/{}/netG_{}.model'.format(args.model_name, epoch),
                "netG_{}.model".format(epoch)
            )

            bucket_manager.upload_file(
                'text-captions/{}/netD_{}.model'.format(args.model_name, epoch),
                "netD_{}.model".format(epoch)
            )

        print("Epoch {} time: {}".format(epoch, time.time() - start))


if __name__ == '__main__':
    print("Starting the main process....", flush=True)
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
    torch.multiprocessing.set_start_method('spawn')
    print("Starting the main process....", flush=True)
    main()
