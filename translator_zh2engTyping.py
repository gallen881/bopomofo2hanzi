from utils import zh2bopomofo, bopomofo2engTyping

while True:
    print(bopomofo2engTyping(''.join(zh2bopomofo(input('?:')))))