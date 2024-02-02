import json
import random
import time
import threading
import pickle
# import numpy as np
import tensorflow as tf
# import evaluate
import re

from pypinyin import lazy_pinyin, Style, load_phrases_dict
load_phrases_dict(json.load(open('pinyin_fix.json', encoding='utf8')))

split_char = '⫯'
punctuations = '、，。？！：；'


# import transformers
# model_name = 'peterhsu/marian-finetuned-kde4-en-to-zh_TW'
# translator = transformers.pipeline('translation', model=model_name, tokenizer=model_name)

# def en2zh_translator(text: str) -> str:
#     return translator(text)[0]['translation_text']

def engTyping_insert_split_char(sentence: str, split_char: str) -> str:
    insert_times = 0
    sentence_list = list(sentence)
    for i, char in enumerate(sentence):
        if char in ' 6347' + punctuations:
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
        sen.append(table.get(char, char))
    return ''.join(sen)

def is_hanzi(char: str) -> bool:
    return '\u4e00' <= char and char <= '\u9fa5'

def is_punctuations(chars: str) -> bool:
    state = True
    for char in chars:
        state = state and (char in punctuations)
        if not state: break
    return state

def del_symbols(text: str) -> str:
    output = ''
    for char in text:
        if is_hanzi(char) or char in punctuations:
            output += char
    return output

def split_zh_en_num(text: str) -> list:
    output = [[], []]
    tmp = ''
    current_type = ''
    for char in text:
        if not char.isalnum() and char != ' ':
            new_type = 'symbol'
        elif is_hanzi(char):
            new_type = 'zh'
        elif char.isdigit():
            try: 
                int(char)
                new_type = 'num'
            except:
                new_type = 'symbol'
        else:
            new_type = 'en'

        if current_type == '':
            current_type = new_type
            tmp = char
        elif new_type == current_type:
            tmp += char
        else:
            output[0].append(tmp)
            output[1].append(current_type)
            tmp = char
            current_type = new_type

    output[0].append(tmp)
    output[1].append(current_type)
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
        if char in ' 6347' + punctuations:
            output += sort(tmp) + char
            tmp = ''
            continue
        tmp += char
    return output

def engTyping_end_fix(text):
    if text != '': return f'{text} ' if text[-1] not in ' 6347' + punctuations else text
    else: return ''

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
        if (IsZhInput(words[-4:]) or words[-1] in punctuations) and IsZhInput(words[-8:-4]):
            return 8
    if len(words) >= 7:
        if (IsZhInput(words[-3:]) or words[-1] in punctuations) and IsZhInput(words[-7:-3]):
            return 7
        elif (IsZhInput(words[-4:]) or words[-1] in punctuations) and IsZhInput(words[-7:-4]):
            return 7
    if len(words) >= 6:
        if (IsZhInput(words[-2:]) or words[-1] in punctuations) and IsZhInput(words[-6:-2]):
            return 6
        elif (IsZhInput(words[-3:]) or words[-1] in punctuations) and IsZhInput(words[-6:-3]):
            return 6
        elif (IsZhInput(words[-4:]) or words[-1] in punctuations) and IsZhInput(words[-6:-4]):
            return 6
    if len(words) >= 5:
        if (IsZhInput(words[-2:]) or words[-1] in punctuations) and IsZhInput(words[-5:-2]):
            return 5
        elif (IsZhInput(words[-3:]) or words[-1] in punctuations) and IsZhInput(words[-5:-3]):
            return 5
    if len(words) >= 4:
        if (IsZhInput(words[-2:]) or words[-1] in punctuations) and IsZhInput(words[-4:-2]):
            return 4
    return 0

def get_data(data: str, amount=0) -> list:
    '''
    `data`: TED or PTT
    '''
    if data == 'TED': path = './datasets/Chinese Traditional.txt'
    elif data == 'PTT': path = './datasets/PTT.txt'
    elif data == 'CPTT': path = './datasets/PTT_clean.txt'
    elif data == 'CPTT20': path = './datasets/PTT_clean_all20.txt'
    elif data == 'WIKI': path = './datasets/wiki.txt'
    elif data == 'CWIKI': path = './datasets/wiki_clean.txt'
    elif data == 'CWIKI20': path = './datasets/wiki_clean_all20.txt'
    else: path = data
    lines = open(path, encoding='utf-8').read().split('\n')
    return lines[:amount] if amount else lines

def zh2bopomofo(zh) -> list:
    output = []
    for pinyin in lazy_pinyin(zh, Style.BOPOMOFO):
        if is_punctuations(pinyin):
            for punc in pinyin:
                output.append(punc)
        else:
            if pinyin[-1] not in 'ˉˊˇˋ˙' + punctuations: pinyin += 'ˉ'
            output.append(pinyin)
    return output

def num2hanzi(num: str) -> str:
    if num == '0' * len(num): return '零' * len(num)
    if len(num) > 28: return ''

    num2hanzi_table = {0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九'}
    units = ['', '萬', '億', '萬億', '兆', '萬兆', '億兆']
    suffix = ['', '十', '百', '千']
    num_list = list(num)
    num_list.reverse()
    splitNumList = []
    l = num_list[0:4]
    num_list = num_list[4:]
    while l:
        splitNumList.append(l)
        l = num_list[0:4]
        num_list = num_list[4:]
    hanzi = ''
    for i, arr in enumerate(splitNumList):
        rst = ''
        for j, digit in enumerate(arr):
            rst = num2hanzi_table[int(digit)] + suffix[j] + rst
        rst += units[i]
        hanzi = rst + hanzi
    
    for item in suffix:
        hanzi = hanzi.replace('零' + item, '零')
    
    for i in range(len(units) - 1, -1, -1):
        val = units[i]
        hanzi = re.sub(r'(零+)' + val, lambda match: '' if len(match.group(1)) == 3 else val, hanzi)
    
    hanzi = re.sub(r'零+', '零', hanzi)
    hanzi = hanzi.replace('個', '')
    hanzi = hanzi.replace('一十', '十')
    hanzi = hanzi.replace('二千', '兩千').replace('二百', '兩百')
    if hanzi.startswith('二萬'): hanzi = '兩萬' + hanzi[2:]
    if hanzi.startswith('二億'): hanzi = '兩億' + hanzi[2:]
    if hanzi.startswith('二兆'): hanzi = '兩兆' + hanzi[2:]
    
    return hanzi
    # https://github.com/xiaoyvning/number2hanzi/blob/master/index.js

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

def get_datasets(tvs, train_pairs, val_pairs, test_pairs, save=[]):
    BATCH_SIZE = 64
    source_vectorization = tvs[0]
    target_vectorization = tvs[1]
    def format_dataset(engTyping, zh):
        engTyping = source_vectorization(engTyping)
        zh = target_vectorization(zh)
        return {'engTyping': engTyping, 'zh': zh[:, :-1]}, zh[:, 1:]

    def make_dataset(pairs):
        engTyping_texts, zh_texts = zip(*pairs)
        engTyping_texts = list(engTyping_texts)
        zh_texts = list(zh_texts)
        dataset = tf.data.Dataset.from_tensor_slices((engTyping_texts, zh_texts)).batch(BATCH_SIZE).map(format_dataset, num_parallel_calls=8)
        return dataset.shuffle(2048).prefetch(16).cache()

    train_dataset = make_dataset(train_pairs)
    val_dataset = make_dataset(val_pairs)
    test_dataset = make_dataset(test_pairs)
    if len(save) == 3:
        train_dataset.save(save[0])
        val_dataset.save(save[1])
        test_dataset.save(save[2])
    # Cannot convert a Tensor of dtype variant to a NumPy array.
        # pickle.dump(train_dataset, open(save[0], 'wb'))
        # pickle.dump(val_dataset, open(save[1], 'wb'))
        # pickle.dump(test_dataset, open(save[2], 'wb'))

    return (train_dataset, val_dataset, test_dataset)

def load_datasets(path):
    return tf.data.Dataset.load(path)

def load_text_vectorization(path):
    from_disk = pickle.load(open(path, "rb"))
    tv = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    tv.set_weights(from_disk['weights'])
    return tv

def get_datasets_and_tv(
        name='',
        data='PTT',
        amount=0,
        vocabulary_size=20000,
        sequence_length=20,
        save_lines=False,
        save_tv=False,
        save_text_pairs=False,
        save_datasets=False,
        load_from_disk=True,
        num_threads=4
    ):
    assert (save_tv or save_lines or save_text_pairs or save_datasets) != load_from_disk, 'save_lines, save_tv and save_text_pair should be different from load_from_disk'
    print('Loading data...')
    if load_from_disk:
        text_pairs = pickle.load(open(f'datasets/{name}_text_pairs.pkl', 'rb'))
        source_vectorization = load_text_vectorization(f'models/{name}_source_vectorization.pkl')
        target_vectorization = load_text_vectorization(f'models/{name}_target_vectorization.pkl')
        train_dataset = load_datasets(f'datasets/{name}_train_dataset.tfrecord')
        val_dataset = load_datasets(f'datasets/{name}_val_dataset.tfrecord')
        test_dataset = load_datasets(f'datasets/{name}_test_dataset.tfrecord')
    else:
        t = time.time()
        lines = get_data(data, amount)
        lines_length = len(lines)
        print('Foramtting data...')
        global deleted_line_count
        deleted_line_count = 0
        errors = []
        def format_lines(tnumber: int, flines: list) -> list:
            global deleted_line_count
            try:
                flines_length = len(flines)
                engTyping_inserted_flines = []
                zh_flines = []
                for i, fline in enumerate(flines):
                    split_type_fline, type_list = split_zh_en_num(fline)
                    for j, _type in enumerate(type_list):
                        if _type == 'num':
                            split_type_fline[j] = num2hanzi(split_type_fline[j])
                        # elif _type == 'en' and split_type_fline[j] != ' ' * len(split_type_fline[j]):
                        #     r = en2zh_translator(split_type_fline[j])
                        #     if is_hanzi(r):
                        #         split_type_fline[j] = r
                        #     else:
                        #         split_type_fline[j] = ''
                        elif _type == 'zh':
                            ...
                        elif split_type_fline[j] in punctuations:
                            ...
                        else:
                            split_type_fline[j] = ''
                    engTyping_inserted_fline = ''.join([bopomofo2engTyping(bpmf) + split_char for bpmf in zh2bopomofo(split_type_fline)])[:-1]
                    process_list[tnumber] = [f'Thread {tnumber}: {i + 1}/{flines_length}', i + 1]
                    if engTyping_inserted_fline == '':
                        deleted_line_count += 1
                        continue
                    engTyping_inserted_flines.append(engTyping_inserted_fline)
                    zh_flines.append('[start]' + split_char + zh_insert_split_char(''.join(split_type_fline), split_char) + split_char + '[end]')
                    # print(f'\rThread {tnumber}: {i}/{flines_length}', end='')
                result_list[tnumber] =(engTyping_inserted_flines, zh_flines)
            except Exception as e:
                errors.append(e)
        threads_list = []
        result_list = [([], [])] * num_threads
        process_list = [['', 0]] * num_threads

        for i in range(num_threads):
            threads_list.append(threading.Thread(target=format_lines, args=(i, lines[i::num_threads])))
            threads_list[-1].start()

        total_p = -1
        while threading.active_count() > 1:
            p = ''
            new_total_p = 0
            for process in process_list:
                p += process[0] + '\n'
                new_total_p += process[1]
            p += f'ETA: {time.strftime("%H:%M:%S", time.gmtime(int(round((lines_length - new_total_p) / (new_total_p - total_p if new_total_p - total_p else 1) * 0.5, 0))))}\n'
            p += f'Total: {new_total_p}/{lines_length}\n'
            p += f'Deleted lines: {deleted_line_count}\n'
            p += f'Errors: {len(errors)}\n'
            total_p = new_total_p
            print('\r' + p, end='')
            time.sleep(0.5)
            for _ in range(num_threads + 4):
                print('\x1b[1A\x1b[2K', end='')
        for thread in threads_list:
            thread.join()
        new_total_p = 0
        for process in process_list:
            print('\r' + process[0])
            new_total_p += process[1]
        print(f'Total: {new_total_p}/{lines_length}')
        print(f'{deleted_line_count} lines deleted.')
        for error in errors:
            print(error)
        print('Generating datasets and text vectorization...')
        engTyping_inserted_lines = []
        zh_lines = []
        for result in result_list:
            engTyping_inserted_lines.extend(result[0])
            zh_lines.extend(result[1])
        if save_lines:
            with open(f'datasets/{name}_engTyping_inserted_lines.txt', 'w', encoding='utf8') as file:
                file.write('\n'.join(engTyping_inserted_lines))
            with open(f'datasets/{name}_zh_lines.txt', 'w', encoding='utf8') as file:
                file.write('\n'.join(zh_lines))
        # engTyping_inserted_lines = [engTyping_insert_split_char(bopomofo2engTyping(''.join(zh2bopomofo(line))), split_char) for line in lines]
        # zh_lines = ['[start]' + split_char + zh_insert_split_char(del_symbols(line), split_char) + split_char + '[end]' for line in lines]
        text_pairs = list(zip(engTyping_inserted_lines, zh_lines))
        if save_text_pairs:
            with open(f'datasets/{name}_text_pairs.pkl', 'wb') as file:
                pickle.dump(text_pairs, file)
        train_pairs, val_pairs, test_pairs = get_data_pairs(text_pairs, shuffle=False)
        print('Data preprocessed.')
        tv_save_names = [f"models/{name}_source_vectorization.pkl", f"models/{name}_target_vectorization.pkl"] if save_tv else []
        source_vectorization, target_vectorization = get_text_vectorization(train_pairs, vocabulary_size, sequence_length, tv_save_names)
        dataset_names = [f"datasets/{name}_train_dataset.tfrecord", f"datasets/{name}_val_dataset.tfrecord", f"datasets/{name}_test_dataset.tfrecord"] if save_datasets else []
        train_dataset, val_dataset, test_dataset = get_datasets((source_vectorization, target_vectorization), train_pairs, val_pairs, test_pairs, dataset_names)
        print('Time used:', time.time() - t)
    return train_dataset, val_dataset, test_dataset, source_vectorization, target_vectorization


def text_classifier(text: str):
    tmp = ''
    engTyping_tmp = ''
    output = []
    for char in text:
        tmp += char
        if IsZhInput(tmp):
            engTyping_tmp += tmp
            tmp = ''
        elif len(tmp) >= 3:
            if engTyping_tmp:
                output.append(engTyping_tmp)
                engTyping_tmp = ''
            if IsZhInput(tmp[-4:]):
                engTyping_tmp = tmp[-4:]
                tmp = ''
            elif IsZhInput(tmp[-3:]):
                engTyping_tmp = tmp[-3:]
                tmp = ''
            elif IsZhInput(tmp[-2:]):
                engTyping_tmp = tmp[-2:]
                tmp = ''

    if not output or engTyping_tmp: output.append(engTyping_tmp)
    return output

def text_classifier2(text: str):
    output = []
    index_6347 = []
    last_index = 0
    for i, char in enumerate(text):
        if char in ' 6347':
            index_6347.append(i)
    for i in index_6347:
        if i < 4:
            for j in range(i):
                print(text[j:i])
                if IsZhInput(text[j:i+1]):
                    output.append(text[j:i+1])
                    last_index = i
                    break
        else:
            for j in range(i - 4, i):
                print(text[j:i])
                if IsZhInput(text[j:i+1]):
                    if j == last_index + 1:
                        output[-1] += text[j:i+1]
                    else:
                        output.append(text[j:i+1])
                    last_index = i
                    break
    return output


