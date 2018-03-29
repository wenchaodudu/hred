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

from model import Embedding, UtteranceEncoder, ContextEncoder, HREDDecoder
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
        decoder = HREDDecoder(word_embedding_dim, hidden_size, hidden_size, len(dictionary)).cuda()
        torch.save(embed, 'embed.pt')
    else:
        print("Using saved parameters.")
        embed = torch.load('embed.pt')
        uenc = torch.load('uenc.pt')
        cenc = torch.load('cenc.pt')
        decoder = torch.load('dec.pt')

    params = list(uenc.parameters()) + list(cenc.parameters()) + list(decoder.parameters())
    # print(params)
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.99)

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
                ave_loss = 0
                # eval on dev


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
            # trg_seqs: (batch_size, max_trg_len)
            cenc_packed_input = pack_padded_sequence(cenc_in, ctc_lengths.cpu().numpy(), batch_first=True)
            cenc_out = cenc(cenc_packed_input)
            # cenc_out: (batch_size, dim2)
            max_len = max(trg_lengths)
            decoder_outputs = Variable(torch.zeros(_batch_size, max_len - 1, len(dictionary))).cuda()
            decoder_hidden = decoder.init_hidden(cenc_out)
            #decoder_input = Variable(torch.LongTensor([dictionary['<start>']] * _batch_size))
            decoder_input = embed(torch.zeros(_batch_size).long().cuda().fill_(dictionary['<start>']))
            for t in range(1, max_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden
                )
                decoder_outputs[:, t-1, :] = decoder_output
                decoder_input = embed(trg_seqs[:, t].cuda())
            
            loss = compute_loss(decoder_outputs, Variable(trg_seqs[:, 1:]).cuda(), Variable(torch.LongTensor(trg_lengths) - 1).cuda())
            ave_loss += loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, 10)
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=1000)
    config = parser.parse_args()
    main(config)
