import tensorflow as tf
import numpy as np
import json
from utils import load_text_vectorization, engTyping_insert_split_char, engTyping_rearrange, engTyping_end_fix
config = json.load(open('config.json', encoding='utf8'))
model_names = []
for i, model_name in enumerate(config['models'].keys()):
    model_names.append(model_name)
    print(f'[{i + 1}]{model_name}')
model_name = model_names[int(input('Please choose a model:')) - 1]
if 'Transformer' in model_name:
    from custom import TransformerEncoder, TransformerDecoder, PositionalEmbedding
    model = tf.keras.models.load_model(f'models/{model_name}', custom_objects={"TransformerEncoder": TransformerEncoder, 'TransformerDecoder': TransformerDecoder, 'PositionalEmbedding': PositionalEmbedding})
else:
    model = tf.keras.models.load_model(f'models/{model_name}')
split_char = config['split_char']
max_decoded_sentence_length = config['models'][model_name]['sequence_length']


source_vectorization = load_text_vectorization(f"models/{model_name[:-6]}_source_vectorization.pkl")
target_vectorization = load_text_vectorization(f"models/{model_name[:-6]}_target_vectorization.pkl")

zh_vocab = target_vectorization.get_vocabulary()
zh_index_lookup = dict(zip(range(len(zh_vocab)), zh_vocab))

def decode_sequence(input_sentence) -> str:
    tokenized_input_sentence = source_vectorization([engTyping_insert_split_char(engTyping_rearrange(engTyping_end_fix(input_sentence)), split_char)])
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
    return decoded_sentence[8:-6].replace(split_char, '')

while True:
    print(decode_sequence(input('?:')))