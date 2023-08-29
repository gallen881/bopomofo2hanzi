import pickle
from sacrebleu import BLEU
bleu = BLEU()

with open('datasets/translator_google_lines.pkl', 'rb') as file:
    engTyping_lines = pickle.load(file)
engTyping_lines = [' '.join(list(line)) for line in engTyping_lines]


split_char = '⫯'
datasets_name = 'PTT_2023_08_06'
with open(f'datasets/{datasets_name}_zh_lines.txt', 'r', encoding='utf-8') as file:
    zh_lines = file.readlines()
lines_len = len(zh_lines)
zh_lines = zh_lines[int(lines_len * 0.85):]
zh_lines = [line.replace('\n', '').replace(split_char, ' ')[8:-6] for line in zh_lines]

assert len(engTyping_lines) == len(zh_lines)
result = bleu.corpus_score(engTyping_lines, [zh_lines])
print(result)