from typing import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import models, Tokenizer
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    RobertaConfig,
    get_linear_schedule_with_warmup,
    AdamW,
    DataCollatorForLanguageModeling,
    pipeline
)
from padding_packing import padding_txt
from torch.nn.utils.rnn import pack_padded_sequence


AVAIL_GPUS = min(1, torch.cuda.device_count())


# https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
# https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb

class RoBERTaDataset(Dataset):
    def __init__(self, txt_dir, max_seq_len, data_collator):
        self.pad_txt, self.attention_mask = padding_txt(txt_dir, max_seq_len=max_seq_len)
        self.data_collator = data_collator
        self.data = self.data_collator.torch_mask_tokens(self.pad_txt) # ->tuple[torch.tensor, torch.tensor]

    def __len__(self):
        return len(self.pad_txt)

    def __getitem__(self, index) -> tuple:
        ids = self.data[0][index]
        target = self.data[1][index]
        att_mask = self.attention_mask[index]

        return ids, att_mask, target


class RoBERTaDataModule(LightningDataModule):
    def __init__(self, data_dir: str = './preprocess_txt/', batch_size: int = 16,
                 max_seq_len: int = 256, vocab_size: int = 50000):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.model_name_or_path = None  # no pretrained weight
        self.data_dir = data_dir
        self.max_seq_length = max_seq_len
        self.vocab_size = vocab_size
        self.train_batch_size = batch_size
        self.eval_batch_size = batch_size
        self.tokenizer = RobertaTokenizerFast.from_pretrained('./tokenizer_'+str(self.vocab_size),
                                                              max_len=self.max_seq_length)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        # self.tokenizer = Tokenizer(models.BPE.from_file('./tokenizer_50000/vocab.json',
        # './tokenizer_50000/merges.txt'))

    def setup(self, stage: Optional[str] = None) -> None:
        train_data_dir = self.data_dir + 'train_data.txt'
        val_data_dir = self.data_dir + 'val_data.txt'
        test_data_dir = self.data_dir + 'test_data.txt'
        self.train_dataset = RoBERTaDataset(train_data_dir, self.max_seq_length, self.data_collator)
        self.val_dataset = RoBERTaDataset(val_data_dir, self.max_seq_length, self.data_collator)
        self.test_dataset = RoBERTaDataset(test_data_dir, self.max_seq_length, self.data_collator)

    def prepare_data(self) -> None:
        train_data_dir = self.data_dir + 'train_data.txt'
        val_data_dir = self.data_dir + 'val_data.txt'
        test_data_dir = self.data_dir + 'test_data.txt'
        Tokenizer(models.BPE.from_file('./tokenizer_50000/vocab.json', './tokenizer_50000/merges.txt'))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                          collate_fn=packing_batch, shuffle=True, num_workers=0)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size,
                          collate_fn=packing_batch, shuffle=False, num_workers=0)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size,
                          collate_fn=packing_batch, shuffle=False, num_workers=0)


def packing_batch(batch) -> tuple:
    length = len(batch)
    ids = torch.stack([batch[i][0] for i in range(length)])
    att_masks = torch.stack([batch[i][1] for i in range(length)])
    lens = torch.LongTensor([batch[0][0].size(0) for _ in range(length)])
    labels = torch.stack([batch[i][2] for i in range(length)])
    pack_pad_tokens = pack_padded_sequence(ids, lengths=lens,
                                           batch_first=True, enforce_sorted=False)

    return ids, att_masks, labels


class RoBERTaTransformer(LightningModule):
    def __init__(self,
                 config,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.0,
                 eval_splits: Optional[list] = None,
                 total_steps: int = 5):
        super(RoBERTaTransformer, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = RobertaForMaskedLM(self.config)
        self.lr = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.eval_splits = eval_splits
        self.total_steps = total_steps

    def forward(self, inputs):
        # inputs:(ids, att_mask, target)
        ids, att_mask, target = inputs
        output = self.model(ids, attention_mask=att_mask, labels=target)
        return output

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs = self.forward(batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outputs = self.forward(batch)
        val_loss, logit = outputs[:2]
        pred = logit.squeeze()
        labels = batch[2]
        self.log('valid_loss', val_loss)
        return {"loss": val_loss, "pred": pred, "labels": labels}

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outputs = self.forward(batch)
        test_loss, logit = outputs[:2]
        pred = logit.squeeze()
        labels = batch[2]
        self.log('test_loss', test_loss)
        return {"loss": test_loss, "pred": pred, "labels": labels}

    def setup(self, stage=None) -> None:
        if stage != 'fit':
            return
        #train_loader = self.trainer.data_module
        #print(train_loader)
        #tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        #ab_size = tb_size * self.trainer.accumulate_grad_batches
        #self.total_steps = int(len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs)

    def configure_optimizers(self):
        """prepare optimizer and schedule(linear warmup and decay)"""
        model_ = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model_.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


if __name__ == '__main__':
    SEED = 42
    seed_everything(SEED)

    vocab_size = 50000
    max_seq_len = 256
    batch_size = 16

    roberta_config = RobertaConfig(
        layer_norm_eps=1e-05,
        max_position_embeddings=256,#512
        num_hidden_layers=6,
        vocab_size=vocab_size,
    )
    data_module = RoBERTaDataModule(data_dir='./preprocess_txt/', batch_size=batch_size,
                                    max_seq_len=max_seq_len, vocab_size=vocab_size)
    data_module.setup('fit')
    model = RoBERTaTransformer(config=roberta_config)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(
        max_epochs=1,
        gpus=AVAIL_GPUS,
        logger=logger,
    )
    trainer.fit(model, datamodule=data_module)
