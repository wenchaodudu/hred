import pdb
import sys
import nltk
import pickle


trg = pickle.load(open('persona.{}.parse.trg'.format(sys.argv[1]), 'rb'))
psn = pickle.load(open('persona.{}.parse.psn'.format(sys.argv[1]), 'rb'))
align = []
for x, res in enumerate(trg):
    if x % 1000 == 0:
        sys.stderr.write(str(x))
    persona = psn[x]
    common = [set(res['words']) & set(p['words']) for p in persona]
    found = -1
    for _, s in enumerate(common):
        parse = persona[_]['parse']
        for w in s:
            ind = parse.index('GEN-{}'.format(w))
            if parse[ind - 1] in ['NT-NN']: 
                found = _
                break
    align.append(found)
    '''
    print(' '.join(res['words']))
    if found > -1:
        print(' '.join(persona[found]['words']))
    else:
        print()
    '''
pickle.dump(align, open('persona.{}.align'.format(sys.argv[1]), 'wb'))
                
            
