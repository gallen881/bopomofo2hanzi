import tensorflow as tf
import json
from utils import get_data, get_data_pairs, get_datasets, get_text_vectorization, zh2bopomofo, bopomofo2engTyping, engTyping_insert_split_char, zh_insert_split_char, del_punctuations
model_name = 'EngTyping2Zh_RNN_PTTTraining.keras'
config = json.load(open('config.json', encoding='utf-8'))
split_char = config['split_char']
VOCABULARY_SIZE = config['models'][model_name]['vocabulary_size']
SEQUENCE_LENGTH = config['models'][model_name]['sequence_length']

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
    latent_dim = 1024

    source = tf.keras.Input(shape=(None,), dtype="int64", name="english")
    x = tf.keras.layers.Embedding(VOCABULARY_SIZE, embed_dim, mask_zero=True)(source)
    encoded_source = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(latent_dim), merge_mode="sum")(x)

    past_target = tf.keras.Input(shape=(None,), dtype="int64", name="spanish")
    x = tf.keras.layers.Embedding(VOCABULARY_SIZE, embed_dim, mask_zero=True)(past_target)
    decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True)
    x = decoder_gru(x, initial_state=encoded_source)
    x = tf.keras.layers.Dropout(0.5)(x)
    target_next_step = tf.keras.layers.Dense(VOCABULARY_SIZE, activation="softmax")(x)
    seq2seq_rnn = tf.keras.Model([source, past_target], target_next_step)

    seq2seq_rnn.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    seq2seq_rnn.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=[tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}', save_best_only=True), tf.keras.callbacks.TensorBoard(log_dir='logs')])
