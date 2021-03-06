import re
from multiprocessing import Pool
import random
from tqdm import tqdm


def preprocessing(x):
    try:
        with open('./text_data/texts' + str(x) + '.txt', 'r') as f: #인코딩 문제 있음
            lines = f.readlines()
            for line in lines:
                text = line.strip()

                text = text.replace('中', '중국')
                text = text.replace('美', '미국')
                text = text.replace('日', '일본')
                text = text.replace('北', '북한')

                pattern = '[^가-힣A-Za-z0-9.,()\'\"\”\‘\“\’·@_/%\s]'  # ( ) ',",@,.,-,\s,_ 제외 제거(special token)
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+)'  # e-mail주소 제거
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '(http|ftp|https)?:\/\/?(?:[-\w.]|(?:\da-fA-F]{2}))+'  # url 제거
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '\d{2,3}[-.\s]\d{3,4}[-.\s]\d{4}'  # 전화번호 제거
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = 'w{3}.\w+.\w+.\w+'  # www.---.---- 제거
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '[A-Za-z0-9_.\s]{1,}.com'  # sweat hankyung.com
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '([가-힣0-9A-Za-z]{1,})?(기자|앵커|뉴스|경제|신문|투데이)(입니다.)?([가-힣]+입니다.)?'
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '([가-힣0-9A-Za-z]+)?(기자|앵커|뉴스|경제|신문|투데이)([가-힣0-9]+)?|(뉴시스)'
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '<[^>]*>'  # html 태그 제거
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '[\r|\n]'  # \r,\n 제거
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '[@_/-]'  # 토큰화할 필요 없는 특수기호
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = '[一-龥ぁ-ゔァ-ヴー々〆〤]'  # 한자 일본어 제거
                text = re.sub(pattern=pattern, repl=' ', string=text)

                pattern = re.compile(r'\s+')  # 이중 space 제거
                text = re.sub(pattern=pattern, repl=' ', string=text)

                text = text.lower()
                #print('check')

                if len(text) > 5:
                    w = open('./preprocess_txt/preprocess2.txt', 'a')
                    w.write(text + '\n')
                    w.close()
    except:
        print('fail')


def train_val_test_split():
    f = open('./preprocess_txt/preprocess1.txt', 'r')
    lines = f.readlines()
    val_lines = []
    test_lines = []
    val_ratio = 20000
    test_ratio = 40000

    for i in range(val_ratio):
        while True:
            r = random.randint(0, len(lines)-1)
            if r not in val_lines:
                f1 = open('./preprocess_txt/val_data.txt', 'a')
                f1.write(lines[r])
                val_lines.append(r)
                f1.close()
                break
    print('finish val')

    for j in range(test_ratio):
        while True:
            r = random.randint(0, len(lines)-1)
            if (r not in val_lines) and (r not in test_lines):
                f2 = open('./preprocess_txt/test_data.txt', 'a')
                f2.write(lines[r])
                test_lines.append(r)
                f2.close()
                break
    print('finish test')

    train_ratio = len(lines) - test_ratio - val_ratio
    for k in range(len(lines)):
        if (k not in val_lines) and (k not in test_lines):
            f3 = open('./preprocess_txt/train_data.txt', 'a')
            f3.write(lines[k])
            f3.close()
            break
    print('finish train')


if __name__ == "__main__":
    num_cores = 8
    pool = Pool(num_cores)
    pool.map(preprocessing, range(0, 557))
    #for i in range(557):
    #    preprocessing(i)
    #    print(i)
    #train_val_test_split()
