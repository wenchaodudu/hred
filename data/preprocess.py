import pdb
import sys


with open(sys.argv[1]) as infile:
    ctc = []
    persona = []
    src_writer = open('{}.src'.format(sys.argv[2]), 'w')
    trg_writer = open('{}.trg'.format(sys.argv[2]), 'w')
    persona_writer = open('{}.psn'.format(sys.argv[2]), 'w')
    for line in infile.readlines():
        sent = line.split('\t')
        first = sent[0].split()
        turn = int(first[0])
        if turn == 1:
            ctc = []
            persona = []
        if line.find('your persona:') > -1:
            persona.append(' '.join(first[3:]))
        else:
            sent[0] = ' '.join(first[1:])
            ctc.append(sent[0])
            context = ' __eou__ '.join(ctc) + ' __eou__'
            src_writer.write(context+'\n')
            if sys.argv[1].find('train') > -1:
                for i, res in enumerate(sent[-1].split('|')):
                    if i == 19:
                        trg_writer.write(res.strip().strip('\n')+'\n')
            else:
                res = sent[-1].split('|')[-1]
                trg_writer.write(res.strip().strip('\n')+'\n')
            persona_writer.write('|'.join(persona) + '\n')
            ctc.append(sent[1])
