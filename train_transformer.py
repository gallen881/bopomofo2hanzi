from utils import get_data, get_data_pairs, get_datasets, get_text_vectorization, zh2bopomofo, bopomofo2engTyping, engTyping_insert_split_char, zh_insert_split_char, del_punctuations
import tensorflow as tf
import json

# model_name = 'EngTyping2Zh_TEDTraining.keras'
model_name = 'EngTyping2Zh_Transformer_PTTTraining.keras'
config = json.load(open('config.json', encoding='utf-8'))
split_char = config['split_char']
VOCABULARY_SIZE = config[model_name]['vocabulary_size']
SEQUENCE_LENGTH = config[model_name]['sequence_length']

from custom import TransformerEncoder, TransformerDecoder, PositionalEmbedding

if __name__ == '__main__':
    print('Loading data...')
    lines = get_data('PTT')
    print('Data loaded.')
    print('Preprocessing data...')
    engTyping_inserted_lines = [engTyping_insert_split_char(bopomofo2engTyping(''.join(zh2bopomofo(line))), split_char) for line in lines]
    zh_lines = ['[start]' + split_char + zh_insert_split_char(del_punctuations(line), split_char) + split_char + '[end]' for line in lines]
    text_pairs = list(zip(engTyping_inserted_lines, zh_lines))
    train_pairs, val_pairs, test_pairs = get_data_pairs(text_pairs)
    print('Data preprocessed.')
    print(train_pairs)
    tv_save_names = [f"models/{model_name[:-6]}_source_vectorization.pkl", f"models/{model_name[:-6]}_target_vectorization.pkl"]
    source_vectorization, target_vectorization = get_text_vectorization(train_pairs, VOCABULARY_SIZE, SEQUENCE_LENGTH, tv_save_names)
    train_dataset, val_dataset, test_dataset = get_datasets((source_vectorization, target_vectorization), train_pairs, val_pairs, test_pairs)


    embed_dim = 256
    dense_dim = 2048
    num_heads = 8

    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="english")
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCABULARY_SIZE, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="spanish")
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCABULARY_SIZE, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    decoder_outputs = tf.keras.layers.Dense(VOCABULARY_SIZE, activation="softmax")(x)
    transformer = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    transformer.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    transformer.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=[tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}', save_best_only=True), tf.keras.callbacks.TensorBoard(log_dir='logs')])

    import numpy as np
    zh_vocab = target_vectorization.get_vocabulary()
    zh_index_lookup = dict(zip(range(len(zh_vocab)), zh_vocab))
    max_decoded_sentence_length = 20
    # transformer = tf.keras.models.load_model(f'models/{model_name}', custom_objects={"TransformerEncoder": TransformerEncoder, 'TransformerDecoder': TransformerDecoder, 'PositionalEmbedding': PositionalEmbedding})

    def decode_sequence(input_sentence):
        tokenized_input_sentence = source_vectorization(engTyping_insert_split_char(input_sentence, split_char))
        decoded_sentence = "[start]"
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = target_vectorization(
                [decoded_sentence])[:, :-1]
            predictions = transformer(
                [tokenized_input_sentence, tokenized_target_sentence])
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = zh_index_lookup[sampled_token_index]
            decoded_sentence += split_char + sampled_token
            if sampled_token == "[end]":
                break
        return decoded_sentence

    while True:
        print(decode_sequence(input('?:')))