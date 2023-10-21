import tensorflow as tf
import numpy as np
from sacrebleu import BLEU
from utils import load_text_vectorization

bleu = BLEU()
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

model_name = 'GRU_CWIKI_2023_09_27_VS20000_SL20_Tue_Oct_17_205851_2023.keras'
split_char = '⫯'
max_decoded_sentence_length = 20

tv_name = 'CWIKI_2023_09_27'
source_vectorization = load_text_vectorization(f"models/{tv_name}_source_vectorization.pkl")
target_vectorization = load_text_vectorization(f"models/{tv_name}_target_vectorization.pkl")

zh_vocab = target_vectorization.get_vocabulary()
zh_index_lookup = dict(zip(range(len(zh_vocab)), zh_vocab))
model = tf.keras.models.load_model(f'models/{model_name}')

datasets_name = 'PTT_2023_08_06'
with open(f'datasets/{datasets_name}_engTyping_inserted_lines.txt', encoding='utf8') as file:
    engTyping_inserted_lines = file.read().split('\n')
with open(f'datasets/{datasets_name}_zh_lines.txt', encoding='utf8') as file:
    zh_lines = file.read().split('\n')
lines_len = len(engTyping_inserted_lines)
assert lines_len == len(zh_lines)
engTyping_inserted_lines = engTyping_inserted_lines[int(lines_len * 0.85):]
zh_lines = zh_lines[int(lines_len * 0.85):]
for i in range(len(zh_lines)):
    zh_lines[i] = ' '.join(zh_lines[i].split(split_char)[1:-1])

pred_sentences = []
loop_times = 200
eng_len = len(engTyping_inserted_lines)
split_point = eng_len // loop_times
for k in range(loop_times + 1):
    lines = engTyping_inserted_lines[k * split_point:(k + 1) * split_point]
    len_lines = len(lines)
    tokenized_input_sentence = source_vectorization(lines)
    decoded_sentences = ["[start]"] * len_lines
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            decoded_sentences)[:, :-1]
        predictions = model(
            [tokenized_input_sentence, tokenized_target_sentence])
        for j in range(len_lines):
            if decoded_sentences[j].endswith('[end]'): continue
            decoded_sentences[j] += split_char + zh_index_lookup[np.argmax(predictions[j, i, :])]
    pred_sentences.extend(decoded_sentences)
    print(f'\r{k}/{loop_times}', end='')


print(pred_sentences)

count = 0
for i in range(len(pred_sentences)):
    pred_sentences[i] = pred_sentences[i][7:]
    if pred_sentences[i].endswith('[end]'): pred_sentences[i] = pred_sentences[i][:-5]
    pred_sentences[i] = pred_sentences[i].replace(split_char, ' ')
    count += 1
    print(f'{count}/{eng_len}', end='\r')

result = bleu.corpus_score(pred_sentences, [zh_lines])
print(result)

len(pred_sentences)

len(zh_lines)