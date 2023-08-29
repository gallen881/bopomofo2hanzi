from utils import load_text_vectorization, engTyping_insert_split_char, engTyping_rearrange, engTyping_end_fix, text_classifier, IsZhInputs
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

def decode_sequence(input_sentences: list) -> str:
    sentences_num = len(input_sentences)
    tokenized_input_sentence = source_vectorization(input_sentences)
    decoded_sentences = ["[start]"] * sentences_num
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            decoded_sentences)[:, :-1]
        predictions = model(
            [tokenized_input_sentence, tokenized_target_sentence])
        for j in range(sentences_num):
            if decoded_sentences[j].endswith('[end]'): continue
            decoded_sentences[j] += split_char + zh_index_lookup[np.argmax(predictions[j, i, :])]
    decoded_sentences
    return decoded_sentences
    # return decoded_sentence.replace(split_char, '')


if __name__ == '__main__':
    while True:
        text = input('?:').lower()
        texts = []
        text_list = text_classifier(text)
        if IsZhInputs(text):
            texts = [engTyping_rearrange(text)]
        elif text_list[0]:
            texts = text_list
        if texts:
            texts = [engTyping_insert_split_char(engTyping_rearrange(engTyping_end_fix(line)), split_char) for line in texts]
        r = decode_sequence(texts)
        print(' '.join([line.replace(split_char, '')[7:-5] for line in r]))