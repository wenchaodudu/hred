import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
from data_loader import get_loader
from model import UtteranceEncoder, ContextEncoder, Decoder


def train(self, input_variable, target_variable,
            encoder, decoder, context, context_hidden,
            encoder_optimizer, decoder_optimizer, criterion,
            last,max_length=None):
    pass

def main(argv):
    train_loader = get_loader('./data/train.src', './data/train.trg', './data/word2id.json', 100) 
    max_length = 30
    UEncoder = UtteranceEncoder(word_vectors, 300)
    CEncoder = ContextEncoder(300, 300, 100)
    decoder = Decoder()
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

       

if __name__ == '__main__':
    main(sys.argv)
