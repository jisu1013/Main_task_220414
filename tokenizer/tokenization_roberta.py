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
    # eos_token: </s>, bos_token: <s>, sep_token: </s>, cls_token: <s>,
    # unk_token: <unk>, pad_token: <pad>, mask_token: <mask>
    tokenizer.train([train_file], trainer)
    tokenizer.model.save(output_dir)
    tokenizer.save('./tokenizer_30522.json')

    return tokenizer


if __name__ == "__main__":
    t1 = time()
    bpe_tokenizer = build_bpe_tokenizer('/home/jisu/PycharmProjects/roBERTa_model/preprocess_txt/preprocess.txt'
                                        , 'tokenizer_30522/', 30522, 2)
    print(time()-t1)

