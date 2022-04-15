import re
from multiprocessing import Pool


def preprocessing(x):
    try:
        with open('./text_data/texts' + str(x) + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                text = line.strip()

                text = text.replace('中', '중국')
                text = text.replace('美', '미국')
                text = text.replace('日', '일본')
                text = text.replace('北', '북한')

                pattern = '[^가-힣A-Za-z0-9.,()\'\e"\”\‘\“\’·@_/%\s]'  # ( ) ',",@,.,-,\s,_ 제외 제거(special token)
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
                print('check')

                if len(text) > 5:
                    w = open('./preprocess_txt/preprocess_3.txt', 'a')
                    w.write(text + '\n')
                    w.close()
    except:
        print('fail')


if __name__ == "__main__":
    num_cores = 8
    #for i in range(0, 556):
    #    preprocessing(i)
    pool = Pool(num_cores)
    pool.map(preprocessing, range(0, 556))