import time
from collections import Counter
import re
import time


def word2idx(words_):
    return {o: i for i, o in enumerate(words_)}


def idx2word(words_):
    return {i: o for i, o in enumerate(words_)}


def generate_vocab(input_, words):
    for i, sentence in enumerate(input_):
        tmp = [x.replace('\'', '').replace(' ', '') for x in sentence[1:-2].split(',')]
        for word in tmp:
            words.update([word])


def convert_vocab(input_):
    words_ = ['_PAD', '_SOS', '_EOS', '_UNK'] #+ words
    idx = word2idx(words_)
    word = idx2word(words_)
    for i, sentence in enumerate(input_):
        input_[i] = [idx[word] if word in idx else 3 for word in sentence]
    with open('./vocab.txt', 'w', encoding='UTF-8') as f:
        for i, v in idx.items():
            f.write(f'{i}:{v}\n')

    return words, input_


if __name__ == '__main__':
    words = Counter()
    t = time.time()
    for i in range(11):
        if i < 10:
            f = open('./preprocess_txt/token_0' + str(i), 'r')
        else:
            f = open('preprocess_txt/token_10', 'r')
        lines = f.readlines()
        generate_vocab(lines, words)
        print(time.time() - t, len(words))
    words_ = {k: v for k, v in words.items() if v > 9}
    words_s = sorted(words_, key=words_.get, reverse=True)
    with open('./vocab1.txt', 'w', encoding='utf-8') as f:
        for w in words_s:
            f.write(str(w) + '\n')
    with open('./vocab_cnt.txt', 'w', encoding='utf-8') as f:
        for k, v in words_:
            f.write(str(k) + '\t' + str(v) + '\n')
    #print(words_)
    print(len(words_))#157526
