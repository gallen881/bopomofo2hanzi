import tensorflow as tf
from custom import *

model_name = 'Transformer_PTT_2023_08_06_VS20000_SL20_Tue_Aug__8_022809_2023.keras'
model = tf.keras.models.load_model(f'models/{model_name}', custom_objects={"TransformerEncoder": TransformerEncoder, 'TransformerDecoder': TransformerDecoder, 'PositionalEmbedding': PositionalEmbedding})
dataset_name = 'PTT_2023_08_06'
test_dataset = tf.data.Dataset.load(f'datasets/{dataset_name}_test_dataset.tfrecord')

print(f'Evaluate: {model_name}')
model.evaluate(test_dataset)