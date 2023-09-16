// var hmm = require('./models/PTT_2023_08_06_engTyping2Zh_HMM100_Mon_Aug__7_201541_2023.json');
var hmm = JSON.parse(hmm)
var start_probability = hmm.start_probability;
var transition_probability = hmm.transition_probability;
var emission_probability = hmm.emission_probability;
// var start_probability = JSON.parse(start_probability);
// var transition_probability = JSON.parse(transition_probability);
// var emission_probability = JSON.parse(emission_probability);
var _default = 0.1e-100;
const punctustions = '、，。？！：；';

function max(arr){
    last = 0;
    max_index = 0;
    for (var i = 0; i < arr.length; i++){
        // console.log(i);
        if (arr[i][0] > last){
            last = arr[i][0];
            max_index = i;
        }
    }
    return max_index;
}

function viterbi(obs, states, start_p, trans_p, emit_p){
    var V = [{}];
    var path = {};

    // Initialize base cases (t == 0)
    for (var i = 0; i < states.length; i++){
        var st = states[i];
        if (st in start_p && st in emit_p){
            if (obs[0] in emit_p[st]){
                V[0][st] = start_p[st] * emit_p[st][obs[0]];
            }else{V[0][st] = _default ** 2}
        }else{V[0][st] = _default ** 2}
        path[st] = [st];
    }

    // Run Viterbi for t > 0
    for (var t = 1; t < obs.length; t++){
        V.push({});
        var newpath = {};

        for (var i = 0; i < states.length; i++){
            var curr_st = states[i];
            path_to_curr_st = [];
            for (let j = 0; j < states.length; j++){
                const prev_st = states[j];
                const trans_prob = trans_p[prev_st]?.[curr_st] || _default;
                const emit_prob = emit_p[curr_st]?.[obs[t]] || _default;
                path_to_curr_st.push([(V[t - 1][prev_st] || _default) * trans_prob * emit_prob, prev_st]);
            }
            // console.log(path_to_curr_st);
            max_index = max(path_to_curr_st);
            var curr_prob = path_to_curr_st[max_index][0];
            var prev_state = path_to_curr_st[max_index][1];
            V[t][curr_st] = curr_prob;
            newpath[curr_st] = path[prev_state] + [curr_st];
        }
        var path = newpath;
    }
    var pl = [];
    // console.log(states.length)
    for (var i = 0; i < states.length; i++){
        var st = states[i];
        // console.log(V[V.length - 1][st], st);
        pl.push([V[V.length - 1][st], st]);
    }
    // console.log(pl);
    max_index = max(pl);
    var prob = pl[max_index][0];
    var end_state = pl[max_index][1];
    return prob, path[end_state];
}

function sort(text){
    var output = '';
    var order = '1qaz2wsxedcrfv5tgbyhnujm8ik,9ol.0p;/-';
    for (var i = 0; i < order.length; i++){
        var char = order[i];
        if (text.includes(char)){
            output += char;
        }
    }
    return output;
}

function engTyping_end_fix(text){
    if (text.length > 0){
        if (!' 6347'.includes(text.slice(-1)) && !punctustions.includes(text.slice(-1))){
            return text + ' ';
        }else{
            return text;
        }
    }else{
        return '';
    }
}

function engTyping_rearrange(text){
    var tmp = '';
    var output = '';
    for (var i = 0; i < text.length; i++){
        var char = text[i];
        if (' 6347'.includes(char) || punctustions.includes(char)){
            output += sort(tmp) + char;
            tmp = '';
            continue;
        }
        tmp += char;
    }
    return output;
}

function decode_sentence(text){
    var tmp = '';
    var observations = [];
    text = engTyping_rearrange(engTyping_end_fix(text.toLowerCase()));
    for (var i = 0; i < text.length; i++){
        var c = text[i];
        tmp += c;
        if (' 6347'.includes(c) || punctustions.includes(c)){
            observations.push(tmp);
            tmp = '';
        }
    }
    var states = [];
    for (var i = 0; i < observations.length; i++){
        var observation = observations[i];
        if (!Object.keys(hmm.engTyping2zh).includes(observation)){
            console.log('Error: unknown observation ' + observation);
        }else{
            states = states.concat(hmm.engTyping2zh[observation]);
        }
    }
    if (states.length == 0){
        return (0, ['']);
    }
    return viterbi(observations, states, start_probability, transition_probability, emission_probability);
}

// let text = 'su3cl3'
// console.log(decode_sentence(text))

const engTyping2zh_button = document.getElementById("engTyping2zh_button");
const engTyping_input_text = document.getElementById("engTyping_input_text");

function translate(){
    var text = engTyping_input_text.value;
    document.getElementById("translated_text").innerHTML = decode_sentence(text);
}

engTyping2zh_button.addEventListener("click", translate);
engTyping_input_text.addEventListener('keypress', function (event){
    if (event.key === 'Enter'){
        event.preventDefault();
        translate();
    }
})