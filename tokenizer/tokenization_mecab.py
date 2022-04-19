from konlpy.tag import Mecab
from threading import Thread
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
from transformers import RobertaTokenizer


mecab = Mecab()


def tokenization(lines_):
    w = open('preprocess_txt/tokenization.txt', 'a')
    for line in tqdm(lines_):
        tokens = mecab.morphs(line)
        #tokens = mecab.pos(line)
        w.write(str(tokens)+'\n')


def check(tokens):
    max_len = 0
    min_len = 0
    avg_len = 0
    total_len = []
    for t in tokens:
        if max_len < len(t):
            max_len = len(t)
        if min_len > len(t):
            min_len = len(t)
        avg_len += len(t)
        total_len.append(len(t))
    return max_len, min_len, (avg_len/(len(tokens)+1e-9)), total_len


if __name__ == "__main__":
    num_cores = 8
    f = open('preprocess_txt/pre_text_3.txt', 'r')
    lines = f.readlines()
    n_lines = len(lines)
    print(n_lines)
    tokenization(lines)
    with Pool(num_cores) as pool:
        pool.map(tokenization, lines)
    print('finish')
