# with open('processed.txt', encoding='utf-8') as file:
#     lines = file.read().split('\n')[:-1]
# engt = []
# zh = []
# for line in lines:
#     tmp, ttmp = line.split('\t')
#     engt.append(tmp)
#     zh.append(ttmp)
from utils import *
import json

print('Loading data...')
lines = get_data('PTT')
print('Preprocessing data...')
engt = [[bopomofo2engTyping(char) for char in zh2bopomofo(line)] for line in lines]
zh = [[char for char in del_punctuations(line)] for line in lines]

start_probability = {}
transition_probability = {}
emission_probability = {}
states = []

print('Calculating probability part 1...')
for engt_line, zh_line in zip(engt, zh):
    # print(engt_line, zh_line)
    start_probability[zh_line[0]] = start_probability.get(zh_line[0], 0) + 1
    tmp = ''
    for engt_char, zh_char in zip(engt_line, zh_line):
        states.append(zh_char) if zh_char not in states else ...
        # print(tmp)
        if tmp != '':
            if tmp not in transition_probability.keys():
                transition_probability[tmp] = {}
            transition_probability[tmp][zh_char] = transition_probability[tmp].get(zh_char, 0) + 1
        tmp = zh_char
        if zh_char not in emission_probability.keys():
            emission_probability[zh_char] = {}
        emission_probability[zh_char][engt_char] = emission_probability[zh_char].get(engt_char, 0) + 1

for key in emission_probability.keys():
    if len(emission_probability[key]) > 1:
        print(key, emission_probability[key])

def sp_calculation(sp: dict):
    tatal = sum(sp.values())
    for key in sp.keys():
        sp[key] /= tatal
    return sp

def tpep_calculation(tpep: dict):
    for key in tpep.keys():
        total = sum(tpep[key].values())
        for key2 in tpep[key].keys():
            tpep[key][key2] /= total
    return tpep

print('Calculating probability part 2...')
start_probability = sp_calculation(start_probability)
transition_probability = tpep_calculation(transition_probability)
emission_probability = tpep_calculation(emission_probability)

print('Saving model...')
json.dump({'start_probability': start_probability, 'transition_probability': transition_probability, 'emission_probability': emission_probability, 'states': states}, open('models/engTyping2Zh_HMM.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
