import tensorflow as tf
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

model_name = 'LSTM_PTT_2023_08_06_VS20000_SL20_Fri_Aug_11_185708_2023.keras'
model = tf.keras.models.load_model(f'models/{model_name}')
# model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy", tfa.metrics.F1Score])
dataset_name = 'PTT_2023_08_06'
test_dataset = tf.data.Dataset.load(f'datasets/PTT_2023_08_06_5length_test_dataset.tfrecord')

print(f'Evaluate: {model_name}')
model.evaluate(test_dataset)