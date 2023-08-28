import json
from sacrebleu import BLEU

# with open('models/PTT_2023_08_06_engTyping2Zh_HMM100_Mon_Aug__7_231753_2023.json', encoding='utf-8') as file:
with open('models/PTT_2023_08_06_engTyping2Zh_HMM70_Fri_Aug_25_164054_2023.json', encoding='utf-8') as file:
    hmm = json.load(file)
start_probability = hmm['start_probability']
transition_probability = hmm['transition_probability']
emission_probability = hmm['emission_probability']

default = 0.1e-100

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # 存儲每個時間步的最大概率
    path = {}  # 存儲每個狀態的最佳路徑

    # 初始化
    for i, st in enumerate(states):
        V[0][st] = start_p.get(st, default) * emit_p.get(st, {}).get(obs[0], default)  # 初始時間步的概率為起始概率乘以發射概率
        path[st] = [st]  # 每個狀態作為自己的最佳路徑的起點

    # 執行 Viterbi 算法當 t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for curr_st in states:
            paths_to_curr_st = []
            for prev_st in states:
                # 計算到達當前狀態的所有路徑的概率
                paths_to_curr_st.append((V[t-1].get(prev_st, default) * trans_p.get(prev_st, {}).get(curr_st, default) * emit_p.get(curr_st, {}).get(obs[t], default), prev_st))
            curr_prob, prev_state = max(paths_to_curr_st)  # 選擇概率最大的路徑
            V[t][curr_st] = curr_prob  # 更新當前時間步和當前狀態的最大概率
            newpath[curr_st] = path[prev_state] + [curr_st]  # 更新最佳路徑

        path = newpath  # 更新最佳路徑字典

    # 輸出 Viterbi 矩陣
    # for line in dptable(V):
    #     print(line)

    # 找到最後一個時間步概率最大的狀態和對應的最佳路徑
    prob, end_state = max([(V[-1][st], st) for st in states])
    return prob, path[end_state]

def decode_sentence(text: str):
    observations = text.split(split_char)
    states = []
    for observation in observations:
        if observation not in hmm['engTyping2zh'].keys():
            print(f'Unknown word: {observation}')
        else:
            states.extend(hmm['engTyping2zh'][observation])
    if states == []:
        return (0, [''])
    return viterbi(observations, states, start_probability, transition_probability, emission_probability)


bleu = BLEU()
split_char = '⫯'
datasets_name = 'PTT_2023_08_06'
with open(f'datasets/{datasets_name}_engTyping_inserted_lines.txt', 'r', encoding='utf-8') as file:
    engTyping_inserted_lines = file.readlines()
lines_len = len(engTyping_inserted_lines)
engTyping_inserted_lines = engTyping_inserted_lines[int(lines_len * 0.85):]
with open(f'datasets/{datasets_name}_zh_lines.txt', 'r', encoding='utf-8') as file:
    zh_lines = file.readlines()
zh_lines = zh_lines[int(lines_len * 0.85):]
assert len(engTyping_inserted_lines) == len(zh_lines)

lines_len = len(engTyping_inserted_lines)
pred_sentences = []
for i in range(len(engTyping_inserted_lines)):
    zh_lines[i] = zh_lines[i].replace('\n', '').replace(split_char, ' ')[8:-6]
    pred_sentences.append(' '.join(decode_sentence(engTyping_inserted_lines[i].replace('\n', ''))[1]))
    print(f'\r{i + 1}/{lines_len}', end='')

result = bleu.corpus_score(pred_sentences, [zh_lines])
print(result)