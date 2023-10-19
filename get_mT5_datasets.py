from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM
import datasets

split_char = '⫯'
data_name = 'CWIKI_2023_09_27'
# model_name = f'mT5_small_PTT_2023_08_06_Thu_Aug_17_081645_2023.ckpt'
# model_name = f'mT5_small_{data_name}_{time.ctime().replace(" ", "_").replace(":", "")}.ckpt'
checkpoint = "google/mt5-small"
prefix = "translate engTyping to Traditional Chinese:".split()

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

with open(f'datasets/{data_name}_engTyping_inserted_lines.txt', 'r', encoding='utf-8') as file:
    engTyping_inserted_lines = file.readlines()
with open(f'datasets/{data_name}_zh_lines.txt', 'r', encoding='utf-8') as file:
    zh_lines = file.readlines()

engTyping_inserted_lines = [prefix + line.split(split_char) for line in engTyping_inserted_lines]
lines_len = len(engTyping_inserted_lines)
train_engTyping_inserted_lines = engTyping_inserted_lines[:int(lines_len * 0.7)]
val_engTyping_inserted_lines = engTyping_inserted_lines[int(lines_len * 0.7):int(lines_len * 0.85)]
test_engTyping_inserted_lines = engTyping_inserted_lines[int(lines_len * 0.85):]
zh_lines = [line.split(split_char)[1:-1] for line in zh_lines]
lines_len = len(zh_lines)
train_zh_lines = zh_lines[:int(lines_len * 0.7)]
val_zh_lines = zh_lines[int(lines_len * 0.7):int(lines_len * 0.85)]
test_zh_lines = zh_lines[int(lines_len * 0.85):]
assert len(train_engTyping_inserted_lines) == len(train_zh_lines)
assert len(val_engTyping_inserted_lines) == len(val_zh_lines)
assert len(test_engTyping_inserted_lines) == len(test_zh_lines)
train_lines = tokenizer(train_engTyping_inserted_lines, text_target=train_zh_lines, max_length=128, is_split_into_words=True, truncation=True)
val_lines = tokenizer(val_engTyping_inserted_lines, text_target=val_zh_lines, max_length=128, is_split_into_words=True, truncation=True)
test_lines = tokenizer(test_engTyping_inserted_lines, text_target=test_zh_lines, max_length=128, is_split_into_words=True, truncation=True)

train_dataset = datasets.Dataset.from_dict(train_lines)
val_dataset = datasets.Dataset.from_dict(val_lines)
test_dataset = datasets.Dataset.from_dict(test_lines)

model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")

# metric = evaluate.load("sacrebleu")

# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [[label.strip()] for label in labels]

#     return preds, labels


tf_train_set = model.prepare_tf_dataset(
    # dataset["train"],
    train_dataset,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

tf_val_set = model.prepare_tf_dataset(
    # dataset["validation"],
    val_dataset,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    # dataset["validation"],
    test_dataset,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

tf_train_set.save(f'datasets/mT5_{data_name}_tf_train_dataset.tfrecord')
tf_val_set.save(f'datasets/mT5_{data_name}_tf_val_dataset.tfrecord')
tf_test_set.save(f'datasets/mT5_{data_name}_tf_test_dataset.tfrecord')