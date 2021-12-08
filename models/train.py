from __future__ import print_function

import argparse
import os
import pickle
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from matplotlib.pyplot import figure
from torch.nn.parallel import DistributedDataParallel
import torch.autograd.profiler as profiler

from torch.distributed.optim import DistributedOptimizer

from models.bcs import BucketManager
from models.model import Generator, Discriminator
from models.training_params import params
from models.utils import weights_init, download_datasets, set_seed


def install(package):
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "cython==0.29.5", "--verbose"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--verbose"])


# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

BUCKET_NAME = "blawml-caseoutcomes"


class ReverseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, embeddings):
        self.dataset = dataset
        self.embeddings = embeddings
        self.words_num = 20

    def __len__(self):
        return len(self.dataset) * 5

    def __getitem__(self, i):
        img, target = self.dataset[int(i / 5)]

        encoding = self.embeddings[target[i % 5]]
        return img, encoding


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
    install("pycocotools==2.0.0")
    print("Starting the main process....", flush=True)
    figure(figsize=(8, 8), dpi=200)
    set_seed(42)
    args = parse_args()

    print("Before environment read....", flush=True)
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    print("Local Rank: {}, Rank: {}".format(args.local_rank, rank), flush=True)

    bucket_manager = BucketManager(BUCKET_NAME)

    print("Initializing torch distributed....", flush=True)
    # Retry init_process_group if worker fails to connect to master
    # due to master host not ready yet
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

    # if use_cuda:
    #     torch.cuda.set_device(rank)

    print("Downloading datasets....", flush=True)
    download_datasets()
    print("Finished downloading dataset", flush=True)

    dataset = dset.CocoCaptions(
        root='coco/train2014',
        annFile='coco/annotations/captions_train2014.json',
        transform=transforms.Compose([
            transforms.Resize(params['image_size']),
            transforms.CenterCrop(params['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    print("Finished preparing dataset object", flush=True)

    pickle_file = open('sbert.pkl', 'rb')
    caption_embeddings = pickle.load(pickle_file)
    dataset = ReverseDataset(dataset, caption_embeddings)

    print("Finished preparing embedding", flush=True)

    print('Number of samples: ', len(dataset), flush=True)
    img, target = dataset[3]

    print("Image Size: ", img.size(), flush=True)
    print(target)

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
    netG = Generator(nz, ngf, nc)

    if (device.type == 'cuda') and (params['ngpu'] > 1) and rank >= 0:
        netG = netG.to(device)
        netG = DistributedDataParallel(
            netG,
            device_ids=[0]
        )
    else:
        netG = DistributedDataParallel(netG)

    print("Finished loading netG DDP".format(rank), flush=True)

    netG.apply(weights_init)

    # Print the model
    print(netG)

    netD = Discriminator(nc, ndf)

    if (device.type == 'cuda') and (params['ngpu'] > 1) and rank >= 0:
        netD.to(device)
        netD = DistributedDataParallel(
            netD,
            device_ids=[0]
        )
    else:
        netD = DistributedDataParallel(netD)

    print("Finished loading netD DDP".format(rank), flush=True)

    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    fake_loss = nn.CosineEmbeddingLoss()

    fixed_noise = torch.randn(32, nz, 1, 1)

    test_input = torch.unsqueeze(torch.unsqueeze(iter(dataloader).next()[1][:32], -1), -1)
    test_input = test_input.type(torch.float)

    fixed_test_embedding = test_input

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(beta1, 0.999))

    real_label = 1.
    fake_label = 0.

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

        def process_iteration():
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            # print("Retrieve data {}".format(rank), flush=True)

            real_cpu = data[0].to(device)
            target = data[1].to(device)
            b_size = real_cpu.size(0)

            # Forward pass real batch through D
            # print("Retrieve netD output {}".format(rank), flush=True)
            output_real = netD(real_cpu, target).view(-1)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # print(real_cpu.shape, target.shape, output_real.shape)

            # Calculate loss on all-real batch
            errD_real = criterion(output_real, label)

            # Calculate gradients for D in backward pass
            # print("Calculate netD loss {}".format(rank), flush=True)
            errD_real.backward()
            # print("Calculate netD loss completed {}".format(rank), flush=True)
            D_x = output_real.mean().item()
            # print("Output mean calculated {}".format(rank), flush=True)

            # Train with all-fake batch
            # Generate batch of latent vectors

            target_input = torch.unsqueeze(torch.unsqueeze(target, -1), -1)
            # print("Debug 1 {}".format(rank), flush=True)
            target_input = target_input.type(torch.float).to(device)
            # print("Debug 2 {}".format(rank), flush=True)

            noise = torch.cat((torch.randn(b_size, nz, 1, 1).to(device), target_input), 1).to(device)
            # Generate fake image batch with G
            # print("Compute netG input {}".format(rank), flush=True)
            fake = netG(noise)
            label.fill_(fake_label)
            # print("real_cpu: {} output:{}".format(noise.shape, fake.shape))

            # Classify all fake batch with D
            # print("Compute netD input {}".format(rank), flush=True)
            output = netD(fake.detach(), target_input).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            # print("Backward errD fake loss {}".format(rank), flush=True)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            # print("Before optimizer step".format(rank), flush=True)
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, target_input).view(-1)

            # Calculate G's loss based on this output
            # print("Before errG loss calculation {}".format(rank), flush=True)
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            # print("After errG loss calculation {}".format(rank), flush=True)
            D_G_z2 = output.mean().item()
            # Update G
            # print("Before optimizer step netG".format(rank), flush=True)
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, params['num_epochs'], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), flush=True)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == params['num_epochs'] - 1) and (i == len(dataloader) - 1)):
                if rank == 0:
                    plt.imshow(
                        np.moveaxis(vutils.make_grid(real_cpu[:64], padding=2, normalize=True).detach().cpu().numpy(),
                                    0, -1),
                        interpolation='nearest'
                    )

                    plt.savefig('images/real_{}_{}.png'.format(i, epoch))
                    bucket_manager.upload_file(
                        'text-captions/{}/images/real_{}_{}.png'.format(args.model_name, i, epoch),
                        'images/real_{}_{}.png'.format(i, epoch)
                    )

                fake = netG(torch.cat((fixed_noise, fixed_test_embedding), 1)).detach().cpu()

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

        #with netD.no_sync():
        #    with netG.no_sync():
                # For each batch in the dataloader
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader, 0):
            process_iteration()

        for i, data in enumerate(dataloader, 0):
            process_iteration()
            break

        print("Epoch {} time: {}".format(epoch, time.time() - start))


if __name__ == '__main__':
    print("Starting the main process....", flush=True)
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
    torch.multiprocessing.set_start_method('spawn')
    print("Starting the main process....", flush=True)
    main()
