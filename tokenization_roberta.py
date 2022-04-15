from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import tokenizers
from time import time


def build_bpe_tokenizer(
        train_file,
        output_dir,
        vocab_size: int,
        min_frequency: int,
):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.train([train_file], trainer)
    tokenizer.model.save(output_dir)

    return tokenizer


if __name__ == "__main__":
    t1 = time()
    bpe_tokenizer = build_bpe_tokenizer('./preprocess_txt/preprocess_2.txt', './output1/', 30000, 2) #50000
    enc = bpe_tokenizer.encode('캐슬린 파커 워싱턴포스트 칼럼니스트 캐슬린 파커 워싱턴포스트 칼럼니스트 도널드 트럼프 미국 대통령이 연말의 들뜬 분위기를 망쳐놓았다.'
                         ,is_pretokenized=False)
    enc_1 = bpe_tokenizer.encode('이스라엘 인 하아레츠의 편집위원실은 트럼프가 “베냐민 네타냐후의 뺨을 후려쳤다”며 미국의 개입은 중동 지역의 게임 규칙을 정하는 데 “러시아에 맞서는 중요한 균형추 역할을 해왔다”고 강조했다.'
                        ,is_pretokenized=False)
    print(enc.ids)
    print(enc.tokens)
    print(enc.n_sequences)
    print(time()-t1)
