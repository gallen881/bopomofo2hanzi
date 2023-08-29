from sacrebleu import BLEU
import pickle

bleu = BLEU()
split_char = '⫯'
datasets_name = 'PTT_2023_08_06'
with open('datasets/translator_google_new_lines.pkl', 'rb') as file:
    translator_google_lines = pickle.load(file)
lines_len = len(translator_google_lines)

with open(f'datasets/{datasets_name}_zh_lines.txt', 'r', encoding='utf-8') as file:
    zh_lines = file.readlines()
zh_lines = zh_lines[len(zh_lines) - lines_len:]
assert len(translator_google_lines) == len(zh_lines)

result = [0, 0]
for i in range(lines_len):
    zh_lines[i] = zh_lines[i].replace('\n', '').split(split_char)[1:-1]
    pred_sentence = list(translator_google_lines[i])
    for j in range(len(zh_lines[i])):
        try: pred_char = pred_sentence[j]
        except:
            result[1] += 1
            continue
        if zh_lines[i][j] == pred_sentence[j]:
            result[0] += 1
        else:
            result[1] += 1
    print(f'\r{i + 1}/{lines_len}', end='')

print(f'\nAccuracy: {result[0] / (result[0] + result[1])}')