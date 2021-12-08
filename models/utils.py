import io
import os
import pickle
import random
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.parallel
import torch.utils.data
import torch.utils.data
import string

from models.bcs import BucketManager

BUCKET_NAME = "outcomes"


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_vectors(file_name):
    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:]).astype(float)
    return data


def download_datasets():
    os.mkdir('coco')
    os.mkdir('images')

    bucket_manager = BucketManager(BUCKET_NAME)
    bucket_manager.download_object('text-captions/train2014.zip', 'coco/train2014.zip')
    bucket_manager.download_object('text-captions/annotations_trainval2014.zip', 'coco/annotations_trainval2014.zip')

    with zipfile.ZipFile('coco/train2014.zip', 'r') as zip_ref:
        zip_ref.extractall('coco/')

    with zipfile.ZipFile('coco/annotations_trainval2014.zip', 'r') as zip_ref:
        zip_ref.extractall('coco/')

    bucket_manager.download_object('sbert-captions.pkl', 'sbert.pkl')


def download_cub_datasets():
    os.mkdir('birds')
    os.mkdir('images')

    bucket_manager = BucketManager(BUCKET_NAME)
    bucket_manager.download_object('text-captions/CUB_200_2011.tgz', 'CUB_200_2011.tgz')
    bucket_manager.download_object('text-captions/birds.zip', 'birds.zip')

    with zipfile.ZipFile('birds.zip', 'r') as zip_ref:
        zip_ref.extractall('birds/')

    with zipfile.ZipFile('birds/birds/text.zip', 'r') as zip_ref:
        zip_ref.extractall('birds/')

    bucket_manager.download_object('sbert-captions-cub.pkl', 'sbert.pkl')


def vectorize_caption(word_to_index, caption, copies=2, debug=False):
    # create caption vector
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    tokens = caption.split(' ')
    cap_v = []
    for t in tokens:
        t = t.strip().encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in word_to_index:
            cap_v.append(word_to_index[t])

    if debug:
        print(tokens, cap_v)

    return cap_v


def word_index():
    x = pickle.load(open('captions.pickle', 'rb'))
    ixtoword = x[2]
    wordtoix = x[3]

    return wordtoix, ixtoword