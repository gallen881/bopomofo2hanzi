import time
import tensorflow as tf
from utils import get_datasets_and_tv

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

# from numba import cuda
# device = cuda.get_current_device()
# device.reset()


dataset_and_tv_name = 'PTT_2023_08_06'
model_name = f'LSTM_{dataset_and_tv_name}_VS20000_SL20_{time.ctime().replace(" ", "_").replace(":", "")}.keras'
split_char = '⫯'
VOCABULARY_SIZE = 20000
SEQUENCE_LENGTH = 20

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset, source_vectorization, target_vectorization = get_datasets_and_tv(dataset_and_tv_name, load_from_disk=True)

    embed_dim = 256
    latent_dim = 1024

    source = tf.keras.Input(shape=(None,), dtype="int64", name="engTyping")
    x = tf.keras.layers.Embedding(VOCABULARY_SIZE, embed_dim, mask_zero=True)(source)
    encoded_source, *encoded_source_state = tf.keras.layers.LSTM(latent_dim, return_state=True)(x)

    past_target = tf.keras.Input(shape=(None,), dtype="int64", name="zh")
    x = tf.keras.layers.Embedding(VOCABULARY_SIZE, embed_dim, mask_zero=True)(past_target)
    decoder_gru = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
    x = decoder_gru(x, initial_state=encoded_source_state)
    x = tf.keras.layers.Dropout(0.5)(x)
    target_next_step = tf.keras.layers.Dense(VOCABULARY_SIZE, activation="softmax")(x)
    seq2seq_rnn = tf.keras.Model([source, past_target], target_next_step)

    seq2seq_rnn.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    seq2seq_rnn.fit(train_dataset, epochs=15, validation_data=val_dataset, callbacks=[tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}', save_best_only=True), tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}_logs')])
    seq2seq_rnn.evaluate(test_dataset)
