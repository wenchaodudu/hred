import json
import pdb
from ast import literal_eval


Preterminal = literal_eval(open('preterminal.txt').readlines()[0])
rule_dict = json.load(open('persona.lex3.dictionary.json'))['rule']

for k, v in rule_dict.items():
    if k not in ['<UNK>', 'RULE: EOD']:
        nts = k.split()
        if nts[-1] in Preterminal[:-3]:
            if any(nt not in Preterminal for nt in nts[1:-1]): 
                print(k)
