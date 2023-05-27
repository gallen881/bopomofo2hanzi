import json
import random
import pickle
import tensorflow as tf

from pypinyin import lazy_pinyin, Style

def engTyping_insert_split_char(sentence: str, split_char: str) -> str:
    insert_times = 0
    sentence_list = list(sentence)
    for i, char in enumerate(sentence):
        if char in ' 6347':
            sentence_list.insert(i + insert_times + 1, split_char)
            insert_times += 1
    return ''.join(sentence_list[:-1])

def zh_insert_split_char(sentence: str, split_char: str) -> str:
    output = ''
    for char in sentence:
        output += char + split_char
    return output[:-1]

def bopomofo2engTyping(sentence: str) -> str:
    sen = []
    with open('bopomofo2engTyping_table.json', encoding='utf8') as file:
        table = json.load(file)
    for char in sentence:
        sen.append(table.get(char, ''))
    return ''.join(sen)

def del_punctuations(text: str) -> str:
    output = ''
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':
            output += char
    return output

def sort(text: str) -> str:
    output = ''
    order = '1qaz2wsxedcrfv5tgbyhnujm8ik,9ol.0p;/-'
    for char in order:
        if char in text:
            output += char
    return output

def engTyping_rearrange(text: str) -> str:
    tmp = ''
    output = ''
    for char in text:
        if char in ' 6347':
            output += sort(tmp) + char
            tmp = ''
            continue
        tmp += char
    return output

def engTyping_end_fix(text):
    return f'{text} ' if text[-1] not in ' 6347' else text

def IsZhInput(words):
    bpmf = [49, 113, 97, 122, 50, 119, 115, 120, 101, 100, 99, 114, 102, 118, 53, 116, 103, 98, 121, 104, 110]
    iwu = [117, 106, 109]
    aouh = [56, 105, 107, 44, 57, 111, 108, 46, 48, 112, 59, 47]
    tone = [32, 54, 51, 52, 55]

    words = [ord(word) for word in words]
    if len(words) == 2:
        if words[0] in [53, 116, 103, 98, 121, 104, 110, 117, 106, 109, 56, 105, 107, 44, 57, 111, 108, 46, 48, 112, 59, 45]:
            if words[1] in tone:
                return True
    if len(words) == 3:
        if (words[0] in bpmf) and (words[1] in iwu + aouh) and (words[2] in tone):
            return True
    if len(words) == 3:
        if (words[0] in iwu) and (words[1] in aouh) and (words[2] in tone):
            return True
    if len(words) == 4:
        if (words[0] in bpmf) and (words[1] in iwu) and (words[2] in aouh) and (words[3] in tone):
            return True
    return False

def IsZhInputs(words: str) -> int:
    if len(words) >= 8:
        if IsZhInput(words[-4:]) and IsZhInput(words[-8:-4]):
            return 8
    if len(words) >= 7:
        if IsZhInput(words[-3:]) and IsZhInput(words[-7:-3]):
            return 7
        elif IsZhInput(words[-4:]) and IsZhInput(words[-7:-4]):
            return 7
    if len(words) >= 6:
        if IsZhInput(words[-2:]) and IsZhInput(words[-6:-2]):
            return 6
        elif IsZhInput(words[-3:]) and IsZhInput(words[-6:-3]):
            return 6
        elif IsZhInput(words[-4:]) and IsZhInput(words[-6:-4]):
            return 6
    if len(words) >= 5:
        if IsZhInput(words[-2:]) and IsZhInput(words[-5:-2]):
            return 5
        elif IsZhInput(words[-3:]) and IsZhInput(words[-5:-3]):
            return 5
    if len(words) >= 4:
        if IsZhInput(words[-2:]) and IsZhInput(words[-4:-2]):
            return 4
    return 0

def get_data(data: str, amount=0) -> list:
    '''
    `data`: TED or PTT
    '''
    if data == 'TED': path = './datasets/Chinese Traditional.txt'
    elif data == 'PTT': path = './datasets/PTT.txt'
    else: path = data
    lines = open(path, encoding='utf-8').read().split('\n')
    return lines[:amount] if amount else lines

def zh2bopomofo(zh) -> list:
    output = []
    for pinyin in lazy_pinyin(zh, Style.BOPOMOFO, errors='ignore'):
        if pinyin[-1] not in 'ˉˊˇˋ˙': pinyin += 'ˉ'
        output.append(pinyin)
    return output

def get_data_pairs(text_pairs, shuffle=True):
    if shuffle: random.shuffle(text_pairs)
    train_pairs_count = int(len(text_pairs) * 0.7)
    val_pairs_count = int((len(text_pairs) - train_pairs_count) / 2)
    train_pairs = text_pairs[:train_pairs_count]                                     # 70% 訓練集
    val_pairs = text_pairs[train_pairs_count:train_pairs_count + val_pairs_count]    # 15% 驗證集
    test_pairs = text_pairs[train_pairs_count + val_pairs_count:]                    # 15% 測試集
    return (train_pairs, val_pairs, test_pairs)

def custom_standardization(tf_str):
    return tf_str
def custom_split(tf_str):
    return tf.strings.split(tf_str, '⫯')

def get_text_vectorization(train_pairs, vocubulary_size, sequence_length, save=[]):
    source_vectorization = tf.keras.layers.TextVectorization(max_tokens=vocubulary_size, output_mode='int', output_sequence_length=sequence_length, standardize=custom_standardization, split=custom_split)
    target_vectorization = tf.keras.layers.TextVectorization(max_tokens=vocubulary_size, output_mode='int', output_sequence_length=sequence_length + 1, standardize=custom_standardization, split=custom_split)

    engTyping_training_texts = [pair[0] for pair in train_pairs]
    zh_training_texts = [pair[1] for pair in train_pairs]
    source_vectorization.adapt(engTyping_training_texts)
    target_vectorization.adapt(zh_training_texts)

    if len(save) == 2:
        pickle.dump({'config': source_vectorization.get_config(), 'weights': source_vectorization.get_weights()}, open(save[0], 'wb'))
        pickle.dump({'config': target_vectorization.get_config(), 'weights': target_vectorization.get_weights()}, open(save[1], 'wb'))

    print(source_vectorization.get_vocabulary())
    print(target_vectorization.get_vocabulary())

    return (source_vectorization, target_vectorization)

def get_datasets(tvs, train_pairs, val_pairs, test_pairs):
    BATCH_SIZE = 64
    source_vectorization = tvs[0]
    target_vectorization = tvs[1]
    def format_dataset(engTyping, zh):
        engTyping = source_vectorization(engTyping)
        zh = target_vectorization(zh)
        return {'english': engTyping, 'spanish': zh[:, :-1]}, zh[:, 1:]

    def make_dataset(pairs):
        engTyping_texts, zh_texts = zip(*pairs)
        engTyping_texts = list(engTyping_texts)
        zh_texts = list(zh_texts)
        dataset = tf.data.Dataset.from_tensor_slices((engTyping_texts, zh_texts)).batch(BATCH_SIZE).map(format_dataset, num_parallel_calls=8)
        return dataset.shuffle(2048).prefetch(16).cache()

    train_dataset = make_dataset(train_pairs)
    val_dataset = make_dataset(val_pairs)
    test_dataset = make_dataset(test_pairs)

    return (train_dataset, val_dataset, test_dataset)

def load_text_vectorization(path):
    from_disk = pickle.load(open(path, "rb"))
    tv = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    tv.set_weights(from_disk['weights'])
    return tv