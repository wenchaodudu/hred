import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json
import pdb
import time
import numpy as np
import argparse
from data_loader import get_loader
from masked_cel import compute_loss

from model import Embedding, UtteranceEncoder, ContextEncoder, HREDDecoder, LatentVariableEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def main(config):
    dictionary = json.load(open('./dictionary.json'))
    vocab_size = len(dictionary)
    word_embedding_dim = 300
    print("Vocabulary size:", len(dictionary))

    print("Loading word vecotrs.")
    word2vec_file = open('./word2vec.vector')
    word_vectors = np.random.uniform(low=-0.25, high=0.25, size=(vocab_size, word_embedding_dim))
    next(word2vec_file)
    for line in word2vec_file:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            word_vectors[dictionary[word]] = np.fromstring(vec, dtype=np.float32, sep=' ')

    train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, 32) 

    hidden_size = 512
    cenc_input_size = hidden_size * 2
    if not config.use_saved:
        embed = Embedding(vocab_size, word_embedding_dim, word_vectors).cuda()
        uenc = UtteranceEncoder(word_embedding_dim, hidden_size).cuda()
        cenc = ContextEncoder(cenc_input_size, hidden_size).cuda()
        context_size = hidden_size * 2 if config.vhred else hidden_size
        decoder = HREDDecoder(word_embedding_dim, context_size, hidden_size, len(dictionary)).cuda()
        torch.save(embed, 'embed.pt')
        if config.vhred:
            prior_enc = LatentVariableEncoder(hidden_size, hidden_size).cuda()
            post_enc = LatentVariableEncoder(hidden_size * 2, hidden_size).cuda()
    else:
        print("Using saved parameters.")
        embed = torch.load('embed.pt')
        uenc = torch.load('uenc.pt')
        cenc = torch.load('cenc.pt')
        decoder = torch.load('dec.pt')
        prior_enc = torch.load('prior.pt')
        post_enc = torch.load('post.pt')

    params = list(uenc.parameters()) + list(cenc.parameters()) + list(decoder.parameters())
    # print(params)
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.99)
    if config.vhred:
        latent_params = list(prior_enc.parameters()) + list(post_enc.parameters())
        latent_optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.99)

    total_loss = 0
    # src_seqs: (N * max_len * word_dim)
    for it in range(10):
        ave_loss = 0
        last_time = time.time()
        for _, (src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_lengths) in enumerate(train_loader):
            if _ % config.print_every_n_batches == 1:
                print(ave_loss / min(_, config.print_every_n_batches), time.time() - last_time)
                last_time = time.time()
                torch.save(uenc, 'uenc.pt')
                torch.save(cenc, 'cenc.pt')
                torch.save(decoder, 'dec.pt')
                if config.vhred:
                    torch.save(prior_enc, 'prior.pt')
                    torch.save(post_enc, 'post.pt')
                ave_loss = 0
            src_seqs = embed(src_seqs.cuda())
            # src_seqs: (N, max_uttr_len, word_dim)
            uenc_packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
            uenc_output = uenc(uenc_packed_input)
            # output: (N, dim1)
            _batch_size = len(ctc_lengths)
            max_len = max(ctc_lengths)
            cenc_in = Variable(torch.zeros(_batch_size, max_len, cenc_input_size).float()).cuda()
            for i in range(len(indices)):
                x, y = indices[i]
                cenc_in[x, y, :] = uenc_output[i]
            # cenc_in: (batch_size, max_turn, dim1)
            ctc_lengths, perm_idx = torch.cuda.LongTensor(ctc_lengths).sort(0, descending=True)
            cenc_in = cenc_in[perm_idx, :, :]
            # cenc_in: (batch_size, max_turn, dim1)
            trg_seqs = trg_seqs.cuda()[perm_idx]
            trg_lengths = Variable(torch.cuda.LongTensor(trg_lengths))[perm_idx]
            # trg_seqs: (batch_size, max_trg_len)
            cenc_packed_input = pack_padded_sequence(cenc_in, ctc_lengths.cpu().numpy(), batch_first=True)
            cenc_out = cenc(cenc_packed_input)
            # cenc_out: (batch_size, dim2)
            max_len = trg_lengths.max().data[0]
            decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, len(dictionary))).cuda()
            if config.vhred:
                sample_prior = prior_enc.sample(cenc_out).cuda()
                decoder_hidden = decoder.init_hidden(torch.stack((cenc_out, sample_prior), dim=1))
            else:
                decoder_hidden = decoder.init_hidden(cenc_out)
            #decoder_input = Variable(torch.LongTensor([dictionary['<start>']] * _batch_size))
            decoder_input = embed(torch.zeros(_batch_size).long().cuda().fill_(dictionary['<start>']))
            for t in range(1, max_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden
                )
                decoder_outputs[:, t-1, :] = decoder_output
                decoder_input = embed(trg_seqs[:, t])
            
            loss = compute_loss(decoder_outputs, Variable(trg_seqs[:, 1:]), trg_lengths - 1)
            ave_loss += loss.data[0]
            if config.vhred:
                trg_lengths, perm_idx = trg_lengths.data.sort(0, descending=True)
                trg_seqs = trg_seqs[perm_idx]
                cenc_out = cenc_out[perm_idx]
                trg_packed = pack_padded_sequence(embed(trg_seqs), trg_lengths.cpu().numpy(), batch_first=True)
                trg_encoded = uenc(trg_packed)
                post_mean, post_var = post_enc(trg_encoded)
                prior_mean, prior_var = prior_enc(cenc_out)
                kl_loss = torch.sum(torch.log(prior_var)) - torch.sum(torch.log(post_var))
                kl_loss += torch.sum((prior_mean - post_mean)**2 / prior_var) 
                kl_loss += torch.sum(post_var / prior_var)
                loss += kl_loss / (2 * _batch_size)

            optimizer.zero_grad()
            if config.vhred:
                latent_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, 10)
            optimizer.step()
            if config.vhred:
                latent_optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=100)
    config = parser.parse_args()
    main(config)
