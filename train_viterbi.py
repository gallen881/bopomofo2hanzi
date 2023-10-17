# from utils import get_data, bopomofo2engTyping, zh2bopomofo, del_symbols
import json
import time


# print('Loading data...')
# lines = get_data('PTT', 100)
# print('Preprocessing data...')
# engt = [[bopomofo2engTyping(char) for char in zh2bopomofo(line)] for line in lines]
# zh = [[char for char in del_symbols(line)] for line in lines]

datasets_name = 'CWIKI_2023_09_27'
model_name = f'{datasets_name}_engTyping2Zh_HMM100_{time.ctime().replace(" ", "_").replace(":", "")}.json'
split_char = '⫯'

engTyping_inserted_lines = open(f'datasets/{datasets_name}_engTyping_inserted_lines.txt', 'r', encoding='utf-8').readlines()
zh_lines = open(f'datasets/{datasets_name}_zh_lines.txt', 'r', encoding='utf-8').readlines()
lines_len = len(engTyping_inserted_lines)
# engt = [line.replace('\n', '').split(split_char) for line in engTyping_inserted_lines]
# zh = [line.replace('\n', '').split(split_char)[1:-1] for line in zh_lines]




start_probability = {}
transition_probability = {}
emission_probability = {}
engTyping2zh = {}
print('Calculating probability part 1...')
t = time.time()
for engt_line, zh_line in zip(engTyping_inserted_lines, zh_lines):
    engt_line = engt_line.replace('\n', '').split(split_char)
    zh_line = zh_line.replace('\n', '').split(split_char)[1:-1]
    # print(engt_line, zh_line)
    if zh_line == []: continue
    start_probability[zh_line[0]] = start_probability.get(zh_line[0], 0) + 1
    tmp = ''
    for engt_char, zh_char in zip(engt_line, zh_line):
        if engt_char not in engTyping2zh.keys():
            engTyping2zh[engt_char] = [zh_char]
        elif zh_char not in engTyping2zh[engt_char]:
            engTyping2zh[engt_char].append(zh_char)
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


print(f'Time used: {time.time() - t} seconds')
print('Saving model...')
json.dump({'start_probability': start_probability, 'transition_probability': transition_probability, 'emission_probability': emission_probability, 'engTyping2zh': engTyping2zh}, open(f'models/{model_name}', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
