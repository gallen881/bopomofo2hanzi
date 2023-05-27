# states = ('你', '好', '啊')
 
observations = []
 
#start_probability = {'你': 0.6, '好': 0.3, '啊': 0.1}
 
#transition_probability = {
#   '你': {'好': 0.7, '啊': 0.2, '你': 0.1},
#   '好': {'好': 0.3, '啊': 0.6, '你': 0.1},
#   '啊': {'啊': 0.8, '好': 0.1, '你': 0.1}
#   }
 
#emission_probability = {
#   '你': {'su3': 1},
 #  '好': {'cl3': 0.6, 'cl4': 0.4},
 #  '啊': {'87': 1},
 #  }
from train_viterbi import start_probability, transition_probability, emission_probability, states

#states = [f'\{hex(i)[1:]}'.decode('utf-8') for i in range(12295, 200415)]


def viterbi(obs, states, start_p, trans_p, emit_p):

    assert obs == observations
    assert states == states
    assert start_p == start_probability
    assert trans_p == transition_probability
    assert emit_p == emission_probability

    assert obs[0] == observations[0]
    assert emit_p[states[0]] == emission_probability[states[0]]
    print(obs[0], type(obs[0]))

    minp_prob = 3.14e-200  # 防止數值下溢

    V = [{}]  # 存儲每個時間步的最大概率
    path = {}  # 存儲每個狀態的最佳路徑

    # 初始化
    for i, st in enumerate(states):
        print(f'count: {i}')
        V[0][st] = start_p.get(st, minp_prob) * emit_p.get(st, {}).get(obs[0], minp_prob)  # 初始時間步的概率為起始概率乘以發射概率
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
                paths_to_curr_st.append((V[t-1].get(prev_st, minp_prob) * trans_p.get(prev_st, {}).get(curr_st, minp_prob) * emit_p.get(curr_st, {}).get(obs[t], minp_prob), prev_st))
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
    tmp = ''
    for c in list(input('?:')):
        tmp += c
        if c in ' 6347':
            observations.append(tmp)
            tmp = ''
    print(observations)
    print(example())