import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json
import pdb
import numpy as np
from data_loader import get_loader
from masked_cel import compute_loss

from model import Embedding, UtteranceEncoder, ContextEncoder, HREDDecoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def main(argv):
    dictionary = json.load(open('./data/dictionary.json'))
    train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, 40) 

    cenc_input_size = 400
    embed = Embedding(len(dictionary), 300).cuda()
    uenc = UtteranceEncoder(300, 200).cuda()
    cenc = ContextEncoder(cenc_input_size, 400).cuda()
    decoder = HREDDecoder(300, 400, 400, len(dictionary)).cuda()

    params = list(uenc.parameters()) + list(cenc.parameters()) + list(decoder.parameters())
    # print(params)
    optimizer = torch.optim.Adam(params, lr=0.0001)

    total_loss = 0
    # src_seqs: (N * max_len * word_dim)
    for _, (src_seqs, src_lengths, indices, trg_seqs, trg_lengths, ctc_lengths) in enumerate(train_loader):
        if _ % 1000 == 0:
            print(_)
        src_seqs = embed(src_seqs.cuda())
        # src_seqs: (N, max_uttr_len, word_dim)
        packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        output = uenc(packed_input)
        # output: (N, dim1)
        _batch_size = len(ctc_lengths)
        max_len = max(ctc_lengths)
        cenc_in = Variable(torch.zeros(_batch_size, max_len, cenc_input_size).float())
        for i in range(len(indices)):
            x, y = indices[i]
            cenc_in[x, y, :] = output[i]
        # cenc_in: (batch_size, max_turn, dim1)
        ctc_lengths, perm_idx = torch.LongTensor(ctc_lengths).sort(0, descending=True)
        cenc_in = cenc_in[perm_idx, :, :]
        # cenc_in: (batch_size, max_turn, dim1)
        trg_seqs = trg_seqs[perm_idx]
        # trg_seqs: (batch_size, max_trg_len)
        packed_input = pack_padded_sequence(cenc_in.cuda(), ctc_lengths.numpy(), batch_first=True)
        cenc_out = cenc(packed_input)
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main(sys.argv)
