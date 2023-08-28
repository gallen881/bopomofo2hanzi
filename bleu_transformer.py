from utils import load_text_vectorization, engTyping_insert_split_char, engTyping_rearrange, engTyping_end_fix
from custom import *
import tensorflow as tf
import numpy as np
import json
from sacrebleu import BLEU
bleu = BLEU()
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

model_name = 'Transformer_PTT_2023_08_06_VS20000_SL20_Tue_Aug__8_022809_2023.keras'
config = json.load(open('config.json', encoding='utf8'))
split_char = '⫯'
max_decoded_sentence_length = 20

tv_name = 'PTT_2023_08_06'
source_vectorization = load_text_vectorization(f"models/{tv_name}_source_vectorization.pkl")
target_vectorization = load_text_vectorization(f"models/{tv_name}_target_vectorization.pkl")

with open(f'datasets/{tv_name}_engTyping_inserted_lines.txt', encoding='utf8') as file:
    engTyping_inserted_lines = file.readlines()
with open(f'datasets/{tv_name}_zh_lines.txt', encoding='utf8') as file:
    zh_lines = file.readlines()
lines_len = len(engTyping_inserted_lines)
assert lines_len == len(zh_lines)
engTyping_inserted_lines = engTyping_inserted_lines[int(lines_len * 0.85):]
zh_lines = zh_lines[int(lines_len * 0.85):]
for i in range(len(zh_lines)):
    zh_lines[i] = ' '.join(zh_lines[i].split(split_char)[1:-1])


zh_vocab = target_vectorization.get_vocabulary()
zh_index_lookup = dict(zip(range(len(zh_vocab)), zh_vocab))
model = tf.keras.models.load_model(f'models/{model_name}', custom_objects={"TransformerEncoder": TransformerEncoder, 'TransformerDecoder': TransformerDecoder, 'PositionalEmbedding': PositionalEmbedding})

def decode_sequence(input_sentence) -> str:
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = model(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = zh_index_lookup[sampled_token_index]
        decoded_sentence += split_char + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence.replace(split_char, ' ')

engTyping_inserted_lines = engTyping_inserted_lines
len_lines = len(engTyping_inserted_lines)
tokenized_input_sentence = source_vectorization(engTyping_inserted_lines)
decoded_sentences = ["[start]"] * len_lines
for i in range(max_decoded_sentence_length):
    tokenized_target_sentence = target_vectorization(
        decoded_sentences)[:, :-1]
    predictions = model(
        [tokenized_input_sentence, tokenized_target_sentence])
    for j in range(len_lines):
        if decoded_sentences[j].endswith('[end]'): continue
        decoded_sentences[j] += split_char + zh_index_lookup[np.argmax(predictions[j, i, :])]


print(decoded_sentences)
for i in range(len(decoded_sentences)):
    decoded_sentences[i] = decoded_sentences[i][7:]
    if decoded_sentences[i].endswith('[end]'): decoded_sentences[i] = decoded_sentences[i][:-5]
    decoded_sentences[i] = decoded_sentences[i].replace(split_char, ' ')


count = 0
# for line in engTyping_inserted_lines:
#     pred_sentence = decode_sequence(line)[7:]
#     if pred_sentence.endswith('[end]'): pred_sentence = pred_sentence[:-5]
#     count += 1
#     print(f'{count}/{len_lines}', end='\r')
result = bleu.corpus_score(decoded_sentences, [zh_lines])
print(result)