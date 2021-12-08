from __future__ import print_function

import argparse
import os
import subprocess
import sys
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
from matplotlib.pyplot import figure
from torch.autograd import Variable
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.bcs import BucketManager
from models.dataset import Cub2011
from models.training_params import params
from models.utils import weights_init, download_cub_datasets, set_seed, word_index, vectorize_caption
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


os.environ['HTTP_PROXY'] = "http://devproxy.bloomberg.com:82"
os.environ['HTTPS_PROXY'] = "http://devproxy.bloomberg.com:82"
os.environ['NO_PROXY'] = "s3.prod.rrdc.bcs.bloomberg.com"


class ImageEncoder(nn.Module):
    def __init__(self, output_token_size):
        super(ImageEncoder, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = torch.nn.Linear(in_features=2048, out_features=output_token_size)

    def forward(self, img):
        img_enc = self.resnet(img)
        return img_enc


class TextEncoder(nn.Module):
    def __init__(self, rnn_type, num_tokens, n_input=300, drop_prob=0.5, n_hidden=128, n_layers=1, bidirectional=True):
        super(TextEncoder, self).__init__()

        self.n_steps = 20
        self.rnn_type = rnn_type

        # size of the dictionary
        self.ntoken = num_tokens

        # size of each embedding vector
        self.ninput = n_input

        # probability of an element to be zeroed
        self.drop_prob = drop_prob

        # Number of recurrent layers
        self.nlayers = n_layers
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # number of features in the hidden state
        self.nhidden = n_hidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob, bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True, enforce_sorted=False)

        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class Generator(nn.Module):

    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + 384, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # input is (nc) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.depth_concatenation = nn.Sequential(
            # state size. (ndf*8)+24 x 4 x 4
            nn.Conv2d(ndf * 16 + 24, ndf * 8, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(ndf * 8, ndf * 4),
            nn.Linear(ndf * 4, 1),
            # state size. (ndf*8) x 4 x 4
            nn.Sigmoid()
        )

    def forward(self, input, embedding):
        x = self.main(input)
        e = torch.reshape(embedding, (embedding.shape[0], 24, 4, 4))
        inp = torch.cat((x, e), 1)

        y = self.depth_concatenation(inp)
        # print(y.shape)
        return y


def install(package):
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
    def __init__(self, dataset, word_to_index, index_to_word):
        self.dataset = dataset
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.words_num = 20
        self.dataset_len = len(self.dataset)

    def __len__(self):
        return len(self.dataset) * 10

    def __getitem__(self, i):
        real_img, target = self.dataset[int(i / 10)]
        fake_img, fake_target = self.dataset[random.randint(0, self.dataset_len-1)]

        captions, cap_lens = self.rotate_caption(target, i)
        fake_captions, fake_cap_lens = self.rotate_caption(fake_target, i)

        return real_img, captions, cap_lens, fake_img, fake_captions, fake_cap_lens

    def rotate_caption(self, target, i):
        captions = vectorize_caption(self.word_to_index, target[i % 10], copies=2)
        captions, cap_lens = self.get_caption(captions)
        j = 1
        while cap_lens == 0:
            captions = vectorize_caption(self.word_to_index, target[((i % 10) + j) % 10], copies=2)
            captions, cap_lens = self.get_caption(captions)
            j += 1

        return captions, cap_lens

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(sent_ix).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.words_num, ), dtype='int64')
        x_len = num_words
        if num_words <= self.words_num:
            x[:num_words] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.words_num]
            ix = np.sort(ix)
            x[:] = sent_caption[ix]
            x_len = self.words_num
        return x, x_len


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
    # install("pycocotools==2.0.0")
    print("Starting the main process....", flush=True)
    figure(figsize=(8, 8), dpi=200)
    set_seed(42)
    args = parse_args()

    torch.autograd.set_detect_anomaly(True)

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

    print("Downloading datasets....", flush=True)
    download_cub_datasets()
    bucket_manager.download_object("text-captions/captions_cub.pickle", "captions.pickle")
    print("Finished downloading dataset", flush=True)

    word_to_index, index_to_word = word_index()

    dataset = Cub2011(
        root="./",
        transform=transforms.Compose([
            transforms.Resize(params['image_size']),
            transforms.CenterCrop(params['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    img, target = dataset[3]

    print(img.shape)
    print(target)

    print("Finished preparing dataset object", flush=True)
    dataset = ReverseDataset(dataset, word_to_index, index_to_word)
    print("Finished preparing embedding", flush=True)

    print('Number of samples: ', len(dataset), flush=True)
    img, captions, cap_lens, fake_img_sample, fake_captions_sample, fake_cap_lens_sample = dataset[3]

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

    # Create the text-encoder
    text_encoder = TextEncoder("GRU", len(word_to_index), n_hidden=384)

    # Create the generator
    netG = Generator(nz, ngf, nc)

    # Create the disciminator
    netD = Discriminator(nc, ndf)

    image_encoder = ImageEncoder(384)
    image_encoder.train()

    if (device.type == 'cuda') and (params['ngpu'] > 1) and rank >= 0:
        image_encoder = image_encoder.to(device)
        image_encoder = DistributedDataParallel(
            image_encoder,
            device_ids=[0]
        )

        text_encoder = text_encoder.to(device)
        text_encoder = DistributedDataParallel(
            text_encoder,
            device_ids=[0]
        )

        netG = netG.to(device)
        netG = DistributedDataParallel(
            netG,
            device_ids=[0]
        )

        netD.to(device)
        netD = DistributedDataParallel(
            netD,
            device_ids=[0]
        )
    else:
        image_encoder = DistributedDataParallel(image_encoder)
        text_encoder = DistributedDataParallel(text_encoder)
        netG = DistributedDataParallel(netG)
        netD = DistributedDataParallel(netD)

    print("Finished loading netG DDP".format(rank), flush=True)
    netG.apply(weights_init)

    # Print the model
    print(text_encoder)
    print(netG)

    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    cosine_loss = nn.CosineEmbeddingLoss()

    fixed_noise = torch.randn(32, nz, 1, 1).to(device)
    fixed_images, fixed_captions, fixed_cap_lens, _, _, _ = iter(dataloader).next()

    print(fixed_captions.shape, fixed_cap_lens.shape, flush=True)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(beta1, 0.999))
    optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=params['lr'])

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

            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            # print("Retrieve data {}".format(rank), flush=True)

            real_cpu = data[0].to(device)
            captions = data[1].to(device)
            cap_lens = data[2].to(device)

            fake_cpu = data[3].to(device)
            fake_captions = data[4].to(device)
            fake_cap_lens = data[5].to(device)

            real_image_embedding = image_encoder(real_cpu)
            fake_image_embedding = image_encoder(fake_cpu)

            b_size = real_cpu.size(0)

            # For the batch of real mappings
            real_hidden = text_encoder.module.init_hidden(b_size)
            real_word_embeddings, real_target = text_encoder(captions, cap_lens, real_hidden)

            # For the batch of real image fake caption mapping
            fake_hidden = text_encoder.module.init_hidden(b_size)
            fake_word_embeddings, fake_target = text_encoder(captions, cap_lens, fake_hidden)

            # Text encoding
            text_encoding_loss = cosine_loss(
                real_image_embedding,
                real_target,
                torch.ones((b_size,), device=device)
            )
            text_encoding_loss += cosine_loss(
                fake_image_embedding,
                real_target,
                torch.zeros((b_size,), device=device)
            )

            text_encoding_loss.backward()
            optimizer_text_encoder.step()

            # Forward pass real batch through D
            output_real = netD(real_cpu, real_target.detach()).view(-1)
            label_real_img = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            criterion(output_real, label_real_img).backward()

            # Forward pass real image fake caption batch through D
            output_real_img = netD(real_cpu, fake_target.detach()).view(-1)
            label_fake_img = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            criterion(output_real_img, label_fake_img).backward()

            # Calculate loss on all-real batch
            errD_real = criterion(output_real, label_real_img) + criterion(output_real_img, label_fake_img)

            # Calculate gradients for D in backward pass
            # print("Calculate netD loss completed {}".format(rank), flush=True)
            D_x_real = output_real.mean().item()
            D_x_real_img_fake_caption = output_real_img.mean().item()

            # print("Output mean calculated {}".format(rank), flush=True)

            # Train with all-fake batch
            # Generate batch of latent vectors

            target_input = torch.unsqueeze(torch.unsqueeze(real_target.detach(), -1), -1)
            target_input = target_input.type(torch.float).to(device)

            noise = torch.cat((torch.randn(b_size, nz, 1, 1).to(device), target_input), 1).to(device)
            # Generate fake image batch with G
            fake = netG(noise)
            generator_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            # print("real_cpu: {} output:{}".format(noise.shape, fake.shape))

            # Classify all fake batch with D
            # print("Compute netD input {}".format(rank), flush=True)
            output = netD(fake.detach(), target_input).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, generator_label)

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
            generator_label.fill_(real_label)
            # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, target_input).view(-1)

            # Calculate G's loss based on this output
            # print("Before errG loss calculation {}".format(rank), flush=True)
            errG = criterion(output, generator_label)
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
                         errD.item(), errG.item(), D_x_real, D_G_z1, D_G_z2), flush=True)

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

                hidden = text_encoder.module.init_hidden(params["batch_size"])
                word_embeddings, target = text_encoder(fixed_captions, fixed_cap_lens, hidden)

                fixed_embedding = torch.unsqueeze(torch.unsqueeze(target, -1), -1)
                fake = netG(torch.cat((fixed_noise, fixed_embedding), 1)).detach().cpu()

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
            torch.save(text_encoder.module.state_dict(), "text_encoder_{}.model".format(epoch))
            torch.save(netG.module.state_dict(), "netG_{}.model".format(epoch))
            torch.save(netD.module.state_dict(), "netD_{}.model".format(epoch))

            bucket_manager.upload_file(
                'text-captions/{}/text_encoder_{}.model'.format(args.model_name, epoch),
                "text_encoder_{}.model".format(epoch)
            )

            bucket_manager.upload_file(
                'text-captions/{}/netG_{}.model'.format(args.model_name, epoch),
                "netG_{}.model".format(epoch)
            )

            bucket_manager.upload_file(
                'text-captions/{}/netD_{}.model'.format(args.model_name, epoch),
                "netD_{}.model".format(epoch)
            )

        with netD.no_sync():
            with netG.no_sync():
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
