# from transformers.keras_callbacks import KerasMetricCallback
from transformers import TFAutoModelForSeq2SeqLM
from transformers import AdamWeightDecay
# from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
# import datasets
import tensorflow as tf
# import keras_nlp
# from nltk.translate.bleu_score import sentence_bleu
# import evaluate
# import numpy as np
import time

# ---

split_char = '⫯'
data_name = 'CWIKI_2023_09_27'
# model_name = f'mT5_small_PTT_2023_08_06_Thu_Aug_17_081645_2023.ckpt'
model_name = f'mT5_small_{data_name}_{time.ctime().replace(" ", "_").replace(":", "")}.ckpt'
checkpoint = "google/mt5-small"

# ---

tokenizer = AutoTokenizer.from_pretrained('models/mT5_pretrained_tokenizer')

# ---

prefix = "translate engTyping to Traditional Chinese:".split()

# ---


# ---

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = TFAutoModelForSeq2SeqLM.from_pretrained('models/mT5_pretrained_model')
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    model.compile(optimizer=optimizer, metrics=["accuracy"])  # No loss argument!

# ---

# model.load_weights('models/mT5_small_PTT_2023_08_06_Thu_Aug_17_081645_2023.ckpt')

# ---


# ---


# ---


# ---

# tf.debugging.disable_traceback_filtering()

# ---

# class BLEU(tf.keras.metrics.Metric):
#     def __init__(self, tokenizer, name="bleu", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.tokenizer = tokenizer
#         self.bleu = sentence_bleu
#         self.bleu_score = self.add_weight(
#             shape=(),
#             initializer='zeros',
#             dtype=self.dtype,
#             name = 'bleu_score'
#         )

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # print(dir(y_true))
#         # print(dir(y_pred))
#         # print(y_true.value_index)
#         # print(y_pred.value_index)
#         # print(y_true[2])
#         # print(y_pred[0])
#         # print(dir(y_true[2]))
#         # print(dir(y_pred[0]))
#         # print(y_true[2].value_index)
#         # print(y_pred[0].value_index)
#         # print(y_true[2][0])
#         # print(y_pred[0][0])
#         # print(y_true[2][0][0])
#         # print(y_pred[0][0][0])
#         # decoded_pred = self.tokenizer.batch_decode(y_pred, skip_special_tokens=True)
#         # y_true = np.where(y_true != -100, y_true, tokenizer.pad_token_id)
#         # decoded_true = self.tokenizer.batch_decode(y_true, skip_special_tokens=True)
#         # decoded_pred, decoded_true = postprocess_text(decoded_pred, decoded_true)
#         result = self.bleu(y_true, y_pred)
#         self.bleu_score.assign(result)

#     def result(self):
#         return self.bleu_score

#     def reset_state(self):
#         self.bleu_score.assign(0.0)

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "tokenizer": self.tokenizer,
#             }
#         )
#         return config
    
# ---

# ---

tf_train_set = tf.data.Dataset.load(f'datasets/mT5_{data_name}_tf_train_dataset.tfrecord')
tf_val_set = tf.data.Dataset.load(f'datasets/mT5_{data_name}_tf_val_dataset.tfrecord')
tf_test_set = tf.data.Dataset.load(f'datasets/mT5_{data_name}_tf_test_dataset.tfrecord')

# ---

# tf.config.run_functions_eagerly(True)

# ---


# metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_val_set)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f'models/{model_name}',save_best_only=True, save_weights_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}_logs')
]

model.fit(x=tf_train_set, validation_data=tf_val_set, epochs=5, callbacks=callbacks)
model.evaluate(tf_test_set)
# from google.colab import drive
# import os
# drive.mount('/content/drive', force_remount=True)
# os.chdir('/content/drive/My Drive/translate')
# model.save_pretrained(f'models/{model_name}')

# ---

# tf.keras.utils.plot_model(model, 'model.png')

# ---

# model.save(f'models/{model_name}', save_format='tf')

# ---

# split_char = '⫯'
# punctuations = '、，。？！：；'

# def engTyping_insert_split_char(sentence: str, split_char: str) -> str:
#     insert_times = 0
#     sentence_list = list(sentence)
#     for i, char in enumerate(sentence):
#         if char in ' 6347' + punctuations:
#             sentence_list.insert(i + insert_times + 1, split_char)
#             insert_times += 1
#     return ''.join(sentence_list[:-1])

# # ---

# inputs = tokenizer(prefix + engTyping_insert_split_char(input('?:'), split_char).split(split_char), is_split_into_words=True, return_tensors="tf", truncation=True).input_ids
# outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))