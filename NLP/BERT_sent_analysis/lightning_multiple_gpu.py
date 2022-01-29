# %%
from typing import Optional

from icecream import ic
import os
import pandas as pd
from datetime import datetime
import warnings
ic(os.getcwd())

import transformers as ppb
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from NLP.BERT_sent_analysis.data_preporation import import collate_fn, ReviewsDataset
from data_preporation import  collate_fn, ReviewsDataset

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class BertClassifierPL(pl.LightningModule):
    def __init__(self, model, params):
        super(BertClassifierPL, self).__init__()
        self.bert = model
        self.params = params

        self.dropout = nn.Dropout(p=params['do'])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.ln1 = nn.BatchNorm1d(768)
        self.fc = nn.Linear(768, params['hidden'])
        self.ln2 = nn.BatchNorm1d(params['hidden'])
        self.fc_out = nn.Linear(params['hidden'], 1)

    def forward(self, inputs, attention_mask):
        # [batch_size, seq_len]
        hidden_states = self.bert(inputs, attention_mask=attention_mask)[0]
        # [batch_size, seq_len, bert_hidden_size]

        hidden_states = hidden_states.mean(dim=1)
        # [batch_size, bert_hidden_size]

        hidden_states = self.fc(self.dropout(self.relu(self.ln1(hidden_states))))
        outputs = self.fc_out(self.ln2(hidden_states))
        # [batch_size, 1]

        # proba = [batch_size, ] - probability to be positive
        return self.sigmoid(outputs)

    def training_step(self, batch, batch_idx):
        inputs, labels, mask = batch['inputs'], batch['labels'], batch['attention_mask']

        output = self(inputs, mask).squeeze()
        loss = F.binary_cross_entropy(output, labels)

        predictions = (output >= 1/2) + 0
        labels = labels.type(torch.cuda.IntTensor)
        acc = accuracy(predictions, labels)

        return dict(loss=loss, acc=acc, batch_size=inputs.shape[0])

    # exactly as training step but can be different
    def validation_step(self, batch, batch_idx):
        inputs, labels, mask = batch['inputs'], batch['labels'], batch['attention_mask']

        output = self(inputs, mask).squeeze()
        loss = F.binary_cross_entropy(output, labels)

        predictions = (output >= 1/2) + 0
        labels = labels.type(torch.cuda.IntTensor)
        acc = accuracy(predictions, labels)

        return dict(val_loss=loss, val_acc=acc, batch_size=inputs.shape[0])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.params['lr'])

        return dict(
            optimizer=optimizer,
            # lr_scheduler=lr_scheduler_config,
        )

    def _aggregate_output(self, loss_key, acc_key, outputs):
        losses, accs, total = [], [], 0
        for x in outputs:
            total += x['batch_size']
            losses.append(x[loss_key] * x['batch_size'])
            accs.append(x[acc_key] * x['batch_size'])
        return dict(loss=torch.stack(losses).sum(), acc=torch.stack(accs).sum(), cnt=total)

    def validation_epoch_end(self, outputs):
        aggregated_output = self._aggregate_output('val_loss', 'val_acc', outputs)
        # now need to get aggregated output from all gpus into one place
        # done with all_gather, ... see https://github.com/PyTorchLightning/pytorch-lightning/discussions/6501#discussioncomment-589529
        # before all_gather aggregated output was a dict of numbers (floats and ints)
        # after all_gather it will be a dict of lists, and list length will be equal to the world size
        aggregated_output = self.all_gather(aggregated_output)
        aggregated_output = {k: sum(val) for k, val in aggregated_output.items()}
        val_acc = aggregated_output['acc'] / aggregated_output['cnt']
        val_loss = aggregated_output['loss'] / aggregated_output['cnt']

        self.logger.experiment.add_scalar('Loss/Val', val_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Acc/Val', val_acc, self.current_epoch)
        self.logger.experiment.add_scalar('Size/Val', aggregated_output['cnt'], self.current_epoch)

        # self.log('val_acc', val_acc, logger=False)
        # self.log('val_loss', val_loss,logger=False)

    # def training_epoch_end(self, outputs):
    #     train_loss, train_acc = self._aggregate_output('loss', 'acc', outputs)
    #     self.logger.experiment.add_scalar('Loss/Train', train_loss, self.current_epoch)
    #     self.logger.experiment.add_scalar('Acc/Train', train_acc, self.current_epoch)


# A DataModule implements 6 key methods:
#   prepare_data (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode, e.g. downloading data only once).
#   setup (things to do on every accelerator in distributed mode) called on every process in DDP.
#   train_dataloader the training dataloader.
#   val_dataloader the val dataloader(s).
#   test_dataloader the test dataloader(s).
#   teardown (things to do on every accelerator in distributed mode when finished, e.g. clean up after fit or test) called on every process in DDP
class ReviewsDataModulePL(pl.LightningDataModule):
    def __init__(self, seed, train_perc, val_perc, tokenizer_class, batch_size, weights):
        super().__init__()
        self.seed = seed
        self.train_perc = train_perc
        self.val_perc = val_perc
        self.tokenizer_class = tokenizer_class
        self.weights = weights
        self.batch_size = batch_size

    def prepare_data(self):
        self.tokenizer_class.from_pretrained(pretrained_weights)

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        reviews_dataset = ReviewsDataset(df[0], tokenizer, df[1])
        torch.manual_seed(self.seed)
        train_size, val_size = int(self.train_perc * len(reviews_dataset)), int(self.val_perc * len(reviews_dataset))
        self.train_data, self.valid_data, self.test_data = random_split(reviews_dataset, [train_size, val_size, len(reviews_dataset) - train_size - val_size])
        ic(len(self.train_data))
        ic(len(self.valid_data))
        ic(len(self.test_data))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=collate_fn)


def version_from_params(p, shortcuts):
    return '__'.join([f'{short}_{p[orig]}' for orig, short in shortcuts.items()])


# %%
if __name__ == '__main__':
    # Todo:
    #   1. Process data only once ... done
    #   2. Logging with multi-gpu
    #   3. Checkpoints with multi-gpu
    #   4.
    #   5.
    # %%
    ic('start')
    warnings.filterwarnings('ignore')
    params = dict(
        train_size=0.8,
        val_size=0.1,
        seed=0xDEAD,
        batch=128,
        hidden=256,
        do=0.5,
        lr=3e-5,
        epochs=40,
        clip=1,
        save_fname='best_bert_sentiment.pt',
    )

    # .......................................................... Loading a pretrained model ........................................................
    # For DistilBERT, Load pretrained model/tokenizer:
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # .......................................................... Creating the dataset ..............................................................
    dataset = ReviewsDataModulePL(
        params['seed'],
        tokenizer_class=tokenizer_class,
        weights=pretrained_weights,
        batch_size=params['batch'],
        train_perc=params['train_size'],
        val_perc=params['val_size']
    )

    # .......................................................... Creating the loggers/callbacks ....................................................
    shortcuts = dict(
        batch='bs',
        hidden='hd',
        do='do',
        lr='lr',
        epochs='ep',
    )
    # time rather should not contain any colon
    time = datetime.now().strftime('%d.%m__%H.%M')
    logs_dir = 'logs'
    experiment_name = 'sent_analysis'
    logger = TensorBoardLogger(logs_dir, name=experiment_name, version=version_from_params(params, shortcuts), sub_dir=time)
    path = os.path.join(logs_dir, experiment_name, version_from_params(params, shortcuts), time, 'checkpoints')

    # save based on valid loss and valid acc
    loss_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=path,
        filename='{epoch:02d}--{val_loss:.2f}',
        mode='min'
    )
    acc_checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=path,
        filename='{epoch:02d}--{val_acc:.2f}',
        mode='max'
    )

    # .......................................................... Model training ............................................................
    # here we init bert outside the BertClassifier for simplicity
    model = model_class.from_pretrained(pretrained_weights)
    pl_model = BertClassifierPL(model, params)

    cb = [loss_checkpoint_callback, acc_checkpoint_callback]
    trainer = pl.Trainer(gpus=2, strategy='ddp', logger=logger, num_sanity_val_steps=0, max_epochs=2, callbacks=cb)
    trainer.fit(pl_model, dataset)
    ic('end')

