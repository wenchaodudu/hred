import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json
import pdb
from data_loader import get_loader

from model import Embedding, UtteranceEncoder, ContextEncoder, HREDDecoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def main(argv):
    dictionary = json.load(open('./data/dictionary.json'))
    train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, 40) 

    '''
    for _, (source, target) in enumerate(train_loader):
        u_encoder_h = UEncoder.init_hidden()
        #encoder_optimizer.zero_grad() # pytorch accumulates gradients, so zero grad clears them up.
        #decoder_optimizer.zero_grad()
        input_length = source.size()[0]
        target_length = target.size()[0]
        u_encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size)).cuda()
        loss = 0
        for ei in range(input_length):
            u_encoder_out, u_encoder_h = UEncoder(source[ei], u_encoder_h)
            u_encoder_outputs[ei] = u_encoder_out[0][0]
        # calculate context
        c_encoder_h = CEncoder.init_hidden()
        c_encoder_out, c_encoder_h = CEncoder(u_encoder_outputs, c_encoder_h)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs,context_hidden)
                loss += criterion(decoder_output[0], target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs,context_hidden)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                # only calculate loss if its the last turn
                if last:
                    loss += criterion(decoder_output[0], target_variable[di])
                if ni == self.EOS_token:
                    break
        if last:
            loss.backward()
        #encoder_optimizer.step()
        #decoder_optimizer.step()
        if last:
            return loss.data[0] / target_length, context_hidden
        else:
            return context_hidden
    '''
    cenc_input_size = 400
    embed = Embedding(len(dictionary), 300).cuda()
    uenc = UtteranceEncoder(300, 200).cuda()
    cenc = ContextEncoder(cenc_input_size, 400).cuda()
    dec = HREDDecoder(300, 400, 400, len(dictionary)).cuda()

    params = list(uenc.parameters()) + list(cenc.parameters()) + list(dec.parameters())
    # print(params)
    optimizer = torch.optim.Adam(params, lr=0.0001)

    '''
    for dialog in train_data:
        total_loss = 0 
        hn = cenc.init_hidden()
        for i in range(len(dialog)-1):
            source, target = dialog[i], dialog[i+1]
            # print(source, target)
            source, target = torch.LongTensor(source).view(1, -1), torch.LongTensor(target).view(1, -1) 
            # print(source, target)
            u_repr = uenc(embed(source))
            # print(u_repr.size())
            c_repr, hn = cenc(u_repr, hn) 
            # print(c_repr.size())
            prdt = dec(c_repr, embed(target))[0]
            # print(prdt.size())
            loss = F.cross_entropy(F.softmax(prdt, dim=1), Variable(target).view(-1))
            # print(loss)
            total_loss += loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()
    '''
    total_loss = 0
    # src_seqs: (N * max_len * word_dim)
    for _, (src_seqs, src_lengths, indices, trg_seqs, ctc_lengths) in enumerate(train_loader):
        src_seqs = embed(src_seqs.cuda())
        # src_seqs: (N * max_len * word_dim)
        packed_input = pack_padded_sequence(src_seqs, src_lengths, batch_first=True)
        packed_output = uenc(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # (N * uenc_out_dim)
        _batch_size = len(ctc_lengths)
        max_len = max(ctc_lengths)
        cenc_in = Variable(torch.zeros(_batch_size, max_len, cenc_input_size).float())
        for i in range(len(indices)):
            x, y = indices[i]
            cenc_in[x, y, :] = output[i, -1, :]
        # cenc_in: (batch_size, max_turn, cenc_in_dim)
        ctc_lengths, perm_idx = torch.LongTensor(ctc_lengths).sort(0, descending=True)
        cenc_in = cenc_in[perm_idx, :, :]
        trg_seqs = trg_seqs[perm_idx.numpy()]
        packed_input = pack_packed_sequence(cenc_in, ctc_lengths.numpy(), batch_first=True)
        packed_output = cenc(cenc_in)
        cenc_out, _ = pad_packed_sequence(packed_output)
        pred = dec(cenc_out, embed(trg_seqs))[0]
        loss = F.cross_entropy(F.softmax(prdt, dim=1), Variable(target).view(-1))
        total_loss += torch.sum(loss.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main(sys.argv)
