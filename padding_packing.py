from abc import ABC

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch
import torch.utils.data as data
from tokenizers import models, Tokenizer
import tokenizers.processors
from tqdm import tqdm


def padding_txt(data_dir, max_seq_len=256, vocab_size=50000):
    f = open(data_dir, 'r')
    lines = f.readlines()
    tokenizer = Tokenizer(models.BPE.from_file('./tokenizer_' + str(vocab_size) + '/vocab.json',
                                               './tokenizer_' + str(vocab_size) + '/merges.txt'))
    tokenizer.post_processor = tokenizers.processors.RobertaProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=max_seq_len)

    pad_tokens = []
    pad_attention = []
    for line in tqdm(lines):
        if line != '\n':
            ids = tokenizer.encode(line).ids
            att_mask = tokenizer.encode(line).attention_mask
            ids = ids + (max_seq_len - len(ids)) * [1]
            att_mask = att_mask + (max_seq_len - len(att_mask)) * [0]
            pad_tokens.append(torch.LongTensor(ids))
            pad_attention.append(torch.LongTensor(att_mask))
    #pad_tokens1 = pad_sequence(pad_tokens, batch_first=True, padding_value=1)
    #pad_attention2 = pad_sequence(pad_attention, batch_first=True, padding_value=0)
    pad_tokens = torch.stack(pad_tokens)
    pad_attention = torch.stack(pad_attention)

    return pad_tokens, pad_attention


if __name__ == '__main__':
    val_pad_data = padding_txt('./preprocess_txt/val_data.txt', 256)
    print('val end')
    #test_pad_data = padding_txt('./preprocess_txt/test_data.txt', 256)
    #print('test end')
    #train_pad_data = padding_txt('./preprocess_txt/preprocess.txt', 256)
    #print('train end')