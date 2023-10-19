from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

checkpoint = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

tokenizer.save_pretrained("models/mT5_pretrained_tokenizer")
model.save_pretrained("models/mT5_pretrained_model")