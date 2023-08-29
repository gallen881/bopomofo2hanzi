from utils import zh2bopomofo, bopomofo2engTyping
from translator_viterbi import decode_sentence

while True:
    text = input('?:')
    decoded_sentence = decode_sentence(bopomofo2engTyping(''.join(zh2bopomofo(text))))[1]
    print(''.join(decoded_sentence))