from utils import get_datasets_and_tv
import tensorflow as tf
import time

dataset_and_tv_name = 'PTT_2023_08_06'
model_name = f'Transformer_{dataset_and_tv_name}_VS20000_SL20_H4_{time.ctime().replace(" ", "_").replace(":", "")}.keras'

VOCABULARY_SIZE = 20000
SEQUENCE_LENGTH = 20

from custom import TransformerEncoder, TransformerDecoder, PositionalEmbedding

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    train_dataset, val_dataset, test_dataset, source_vectorization, target_vectorization = get_datasets_and_tv(dataset_and_tv_name, load_from_disk=True)

    embed_dim = 256
    dense_dim = 2048
    num_heads = 4

    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="engTyping")
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCABULARY_SIZE, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(encoder_outputs)

    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="zh")
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCABULARY_SIZE, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    decoder_outputs = tf.keras.layers.Dense(VOCABULARY_SIZE, activation="softmax")(x)
    transformer = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    transformer.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    transformer.fit(train_dataset, epochs=15, validation_data=val_dataset, callbacks=[tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}', save_best_only=True), tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}_logs')])
    transformer.evaluate(test_dataset)

    print(transformer.summary())
    print(f'Model name: {model_name}')
    print(f'VOCABULARY_SIZE: {VOCABULARY_SIZE}')
    print(f'SEQUENCE_LENGTH: {SEQUENCE_LENGTH}')
    print(f'Heads: {num_heads}')
    print('Done!')