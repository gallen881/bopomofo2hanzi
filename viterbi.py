import json
import time
from utils import engTyping_end_fix, engTyping_rearrange

with open('models/engTyping2Zh_HMM70.json', encoding='utf-8') as file:
    hmm = json.load(file)
start_probability = hmm['start_probability']
transition_probability = hmm['transition_probability']
emission_probability = hmm['emission_probability']

default = 0.1e-100

def viterbi(obs, states, start_p, trans_p, emit_p):

    assert obs == observations
    assert states == states
    assert start_p == start_probability
    assert trans_p == transition_probability
    assert emit_p == emission_probability

    assert obs[0] == observations[0]
    assert emit_p[states[0]] == emission_probability[states[0]]
    print(obs[0], type(obs[0]))

    V = [{}]  # 存儲每個時間步的最大概率
    path = {}  # 存儲每個狀態的最佳路徑

    # 初始化
    for i, st in enumerate(states):
        print(f'count: {i}')
        V[0][st] = start_p.get(st, default) * emit_p.get(st, {}).get(obs[0], default)  # 初始時間步的概率為起始概率乘以發射概率
        path[st] = [st]  # 每個狀態作為自己的最佳路徑的起點

    # 執行 Viterbi 算法當 t > 0
    for t in range(1, len(obs)):
        print(obs[t])

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
    for line in dptable(V):
        print(line)

    # 找到最後一個時間步概率最大的狀態和對應的最佳路徑
    prob, end_state = max([(V[-1][st], st) for st in states])
    return prob, path[end_state]


def dptable(V):
    # 輸出表格的步驟
    yield ' ' * 4 + '    '.join(states)  # 輸出狀態集合

    for t in range(len(V)):
        # 輸出每一行的概率值
        yield '{}   '.format(t) + '    '.join(['{:.4f}'.format(V[t][state]) for state in V[0]])


def example():
    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)

while True:
    _t = time.time()
    tmp = ''
    observations = []
    for c in list(engTyping_rearrange(engTyping_end_fix(input('?:')))):
        tmp += c
        if c in ' 6347':
            observations.append(tmp)
            tmp = ''
    states = []
    for observation in observations:
        if observation not in hmm['engTyping2zh'].keys():
            print(f'Unknown word: {observation}')
            states = []
            break
        states.extend(hmm['engTyping2zh'][observation])
    if states == []:
        continue
    print(example())
    print(observations)
    print(f'total time: {time.time() - _t}')