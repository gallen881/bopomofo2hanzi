from utils import load_text_vectorization, engTyping_insert_split_char, engTyping_rearrange, engTyping_end_fix
import tensorflow as tf
import numpy as np

model_name = 'LSTM_PTT_2023_08_06_VS20000_SL20_Fri_Aug_11_185708_2023.keras'
split_char = '⫯'
max_decoded_sentence_length = 20

tv_name = 'PTT_2023_08_06'
source_vectorization = load_text_vectorization(f"models/{tv_name}_source_vectorization.pkl")
target_vectorization = load_text_vectorization(f"models/{tv_name}_target_vectorization.pkl")

zh_vocab = target_vectorization.get_vocabulary()
zh_index_lookup = dict(zip(range(len(zh_vocab)), zh_vocab))
model = tf.keras.models.load_model(f'models/{model_name}')

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
    return decoded_sentence
    # return decoded_sentence.replace(split_char, '')


if __name__ == '__main__':
    while True:
        print(decode_sequence(input('?:')))