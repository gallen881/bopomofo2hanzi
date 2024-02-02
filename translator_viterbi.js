// var hmm = require('./models/PTT_2023_08_06_engTyping2Zh_HMM100_Mon_Aug__7_201541_2023.json');
var hmm = JSON.parse(hmm)
var start_probability = hmm.start_probability;
var transition_probability = hmm.transition_probability;
var emission_probability = hmm.emission_probability;
// var start_probability = JSON.parse(start_probability);
// var transition_probability = JSON.parse(transition_probability);
// var emission_probability = JSON.parse(emission_probability);
var _default = 0.1e-100;
const punctuations = '、，。？！：；';

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
    return (prob, path[end_state]);
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
        if (!' 6347'.includes(text.slice(-1)) && !punctuations.includes(text.slice(-1))){
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
        if (' 6347'.includes(char) || punctuations.includes(char)){
            output += sort(tmp) + char;
            tmp = '';
            continue;
        }
        tmp += char;
    }
    return output;
}

function engTyping_insert_split_char(sentence, split_char) {
    let insert_times = 0;
    let sentence_list = Array.from(sentence);
    for (let i = 0; i < sentence.length; i++) {
      if (sentence[i] === ' ' || punctuations.includes(sentence[i])) {
        sentence_list.splice(i + insert_times + 1, 0, split_char);
        insert_times++;
      }
    }
    return sentence_list.slice(0, -1).join('');
  }

// function text_classifier(text) {
//     let tmp = '';
//     let engTyping_tmp = '';
//     let output = [];
//     for (let char of text) {
//       tmp += char;
//       if (IsZhInput(tmp)) {
//         engTyping_tmp += tmp;
//         tmp = '';
//       } else if (tmp.length >= 3) {
//         if (engTyping_tmp) {
//           output.push(engTyping_tmp);
//           engTyping_tmp = '';
//         }
//         if (IsZhInput(tmp.slice(-4))) {
//           engTyping_tmp = tmp.slice(-4);
//           tmp = '';
//         } else if (IsZhInput(tmp.slice(-3))) {
//           engTyping_tmp = tmp.slice(-3);
//           tmp = '';
//         } else if (IsZhInput(tmp.slice(-2))) {
//           engTyping_tmp = tmp.slice(-2);
//           tmp = '';
//         }
//       }
//     }
//     if (!output.length || engTyping_tmp) output.push(engTyping_tmp);
//     return output;
//   }

function text_classifier(text) {
  let output = [];
  let index_6347 = [];
  let last_index = 0;
  for (let i = 0; i < text.length; i++) {
    let char = text[i];
    if (char === ' ' || char === '6' || char === '3' || char === '4' || char === '7') {
      index_6347.push(i);
    }
  }
  for (let i of index_6347) {
    if (i < 4) {
      for (let j = 0; j < i; j++) {
        let substring = text.slice(j, i+1);
        console.log(substring);
        if (IsZhInput(substring)) {
          output.push(substring);
          last_index = i;
          break;
        }
      }
    } else {
      for (let j = i - 4; j < i; j++) {
        let substring = text.slice(j, i+1);
        console.log(substring);
        if (IsZhInput(substring)) {
          if (j === last_index + 1) {
            output[output.length - 1] += substring;
          } else {
            output.push(substring);
          }
          last_index = i;
          break;
        }
      }
    }
  }
  console.log(output);
  return output;
}

  function IsZhInput(words) {
    const bpmf = [49, 113, 97, 122, 50, 119, 115, 120, 101, 100, 99, 114, 102, 118, 53, 116, 103, 98, 121, 104, 110];
    const iwu = [117, 106, 109];
    const aouh = [56, 105, 107, 44, 57, 111, 108, 46, 48, 112, 59, 47];
    const tone = [32, 54, 51, 52, 55];
  
    words = Array.from(words).map((word) => word.charCodeAt(0));
    if (words.length === 2) {
      if (
        [53, 116, 103, 98, 121, 104, 110, 117, 106, 109, 56, 105, 107, 44, 57, 111, 108, 46, 48, 112, 59, 45].includes(
          words[0]
        ) &&
        tone.includes(words[1])
      ) {
        return true;
      }
    }
    if (words.length === 3) {
      if (
        bpmf.includes(words[0]) &&
        (iwu.concat(aouh).includes(words[1])) &&
        tone.includes(words[2])
      ) {
        return true;
      }
    }
    if (words.length === 3) {
      if (iwu.includes(words[0]) && aouh.includes(words[1]) && tone.includes(words[2])) {
        return true;
      }
    }
    if (words.length === 4) {
      if (
        bpmf.includes(words[0]) &&
        iwu.includes(words[1]) &&
        aouh.includes(words[2]) &&
        tone.includes(words[3])
      ) {
        return true;
      }
    }
    return false;
  }
  
  function IsZhInputs(words) {
  
    if (words.length >= 8) {
      if (
        (IsZhInput(words.slice(-4)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-8, -4))
      ) {
        return 8;
      }
    }
    if (words.length >= 7) {
      if (
        (IsZhInput(words.slice(-3)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-7, -3))
      ) {
        return 7;
      } else if (
        (IsZhInput(words.slice(-4)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-7, -4))
      ) {
        return 7;
      }
    }
    if (words.length >= 6) {
      if (
        (IsZhInput(words.slice(-2)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-6, -2))
      ) {
        return 6;
      } else if (
        (IsZhInput(words.slice(-3)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-6, -3))
      ) {
        return 6;
      } else if (
        (IsZhInput(words.slice(-4)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-6, -4))
      ) {
        return 6;
      }
    }
    if (words.length >= 5) {
      if (
        (IsZhInput(words.slice(-2)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-5, -2))
      ) {
        return 5;
      } else if (
        (IsZhInput(words.slice(-3)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-5, -3))
      ) {
        return 5;
      }
    }
    if (words.length >= 4) {
      if (
        (IsZhInput(words.slice(-2)) || punctuations.includes(words.slice(-1))) &&
        IsZhInput(words.slice(-4, -2))
      ) {
        return 4;
      }
    }
    return 0;
  }

function decode_sentence(text){
    var tmp = '';
    var observations = [];
    for (var i = 0; i < text.length; i++){
        var c = text[i];
        tmp += c;
        if (' 6347'.includes(c) || punctuations.includes(c)){
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

function translator(textt){
    let texts = [];
    let text = textt.toLowerCase();
    let text_list = text.length >= 4 ? text_classifier(text) : [''];
    console.log(text_list)
    // if (IsZhInputs(text)) {
    //   texts = [engTyping_rearrange(text)];
    // } else 
    //  {
    //   texts = text_list;
    // }
    texts = text_list
    let output = [];
    if (texts.length) {
      for (let text of texts) {
        console.log(text)
        let r = decode_sentence(text);
        console.log(r);
        // if (r[0] > 0) {
        //   output.push(r[1].join(''));
        // }
        output.push(r);
      }
    }
    let translated_msg = output.join(' ');
    return translated_msg;
}

// let text = 'su3cl3'
// console.log(decode_sentence(text))

const engTyping2zh_button = document.getElementById("engTyping2zh_button");
const engTyping_input_text = document.getElementById("engTyping_input_text");

function translate(){
    var text = engTyping_input_text.value;
    document.getElementById("translated_text").innerHTML = translator(text);
}

engTyping2zh_button.addEventListener("click", translate);
engTyping_input_text.addEventListener('keypress', function (event){
    if (event.key === 'Enter'){
        event.preventDefault();
        translate();
    }
})