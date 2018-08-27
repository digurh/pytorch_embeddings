import os
import sys
import re
import argparse
from tqdm import tqdm
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



torch.manual_seed(0)

parse = argparse.ArgumentParser()
parse.add_argument('--corpus_loc', default='../data/elliot.txt')
parse.add_argument('--step_size', type=int, default=50)
parse.add_argument('--batch_size', type=int, default=1)
parse.add_argument('--hidden_size', type=int, default=64)
parse.add_argument('--window_size', type=int, default=2)
parse.add_argument('--embed_dim', type=int, default=100)
parse.add_argument('--n_hidden', type=int, default=128)
parse.add_argument('--n_epochs', type=int, default=10)


locals().update(vars(parse.parse_args()))


with open(corpus_loc, 'r') as f:
    corpus_raw = list(f.read())

corpus = []
str_bldr = ''
includes = [',', '.', ';', '-', '\'', '\"', '?', '!', '\n', ':']

for c in corpus_raw:
    if c.isalpha():
        str_bldr = ''.join([str_bldr, c])
    else:
        if str_bldr is not '':
            corpus.append(str_bldr.lower())
            str_bldr = ''
        if c in includes:
            corpus.append(c)

count = Counter()
for word in corpus:
    count[word] += 1

vocabulary = list(set([word for word in corpus]))

word_idx = {word: i for i, word in enumerate(vocabulary)}
idx_word = {i: word for i, word in enumerate(vocabulary)}

ngrams = []

for w in range(len(corpus)):
    wi = word_idx[corpus[w]]
    context = []
    for c in range(w-window_size, w+window_size+1):
        if c < 0 or c > len(corpus)-1:
            context.append(0)
        elif c == w:
            pass
        else:
            con_word = word_idx[corpus[c]]
            context.append(con_word)
    ngrams.append((wi, context))


class NGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, window_size, n_hidden=128):
        super(NGramModel, self).__init__()
        self.window_size = window_size
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.l1 = nn.Linear(self.window_size * embed_dim, n_hidden)
        self.l2 = nn.Linear(n_hidden, vocab_size)

    def forward(self, X):
        embeds = self.embeddings(X).view((1, -1))
        X = F.relu(self.l1(embeds))
        X = self.l2(X)
        log_probs = F.log_softmax(X, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
network = NGramModel(len(vocabulary), embed_dim, window_size*2)
opt = optim.SGD(network.parameters(), lr=0.001)

print('Begin training embedding')

for epoch in tqdm(range(n_epochs)):
    total_loss = 0
    for target, context in ngrams:
        # get context for a given target and turn it into LongTensor
        context = torch.tensor(context, dtype=torch.long)

        # clear gradients since pytorch accumulates gradients
        network.zero_grad()

        # run the forward pass on the network
        log_probs = network.forward(context)

        # compute loss: target must be Tensor
        loss = loss_function(log_probs, torch.tensor([target], dtype=torch.long))

        # backward pass and gradient update
        loss.backward()
        opt.step()

        total_loss += loss.item()
    losses.append(total_loss)

print(losses)
