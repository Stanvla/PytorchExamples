# %%
import os
import warnings
from datetime import datetime
from typing import Optional

from icecream import ic
import pandas as pd
import transformers as ppb

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from pytorch_lightning.profiler import PyTorchProfiler

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data_preporation import collate_fn, ReviewsDataset


class BertClassifierPL(pl.LightningModule):
    def __init__(self, model, params):
        super(BertClassifierPL, self).__init__()
        self.bert = model
        self.params = params

        self.dropout = nn.Dropout(p=params['do'])
        self.relu = nn.ReLU()

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
        return outputs

    def _perform_step(self, batch):
        inputs, labels, mask = batch['inputs'], batch['labels'], batch['attention_mask']
        output = self(inputs, mask).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, labels)
        predictions = (output >= 1 / 2) + 0
        labels = labels.type(torch.cuda.IntTensor)
        acc = accuracy(predictions, labels)
        # return {loss_key: loss, acc_key: acc, batch: inputs.shape[0]}
        return loss, acc, inputs.shape[0]

    def training_step(self, batch, batch_idx):
        loss, acc, batch_size = self._perform_step(batch)
        return dict(loss=loss, acc=acc, batch_size=batch_size)

    # exactly as training step but can be different
    def validation_step(self, batch, batch_idx):
        loss, acc, batch_size = self._perform_step(batch)
        return dict(val_loss=loss, val_acc=acc, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        loss, acc, batch_size = self._perform_step(batch)
        return dict(test_loss=loss, test_acc=acc, batch_size=batch_size)

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

    def _epoch_end(self, outputs, loss_key, acc_key, name):
        # name can be {Train, Test, Val}
        aggregated_output = self._aggregate_output(loss_key, acc_key, outputs)
        # now need to get aggregated output from all gpus into one place
        # done with all_gather, ... see https://github.com/PyTorchLightning/pytorch-lightning/discussions/6501#discussioncomment-589529
        # before all_gather aggregated output was a dict of numbers (floats and ints)
        # after all_gather it will be a dict of lists, and list length will be equal to the world size
        aggregated_output = self.all_gather(aggregated_output)
        aggregated_output = {k: sum(val) for k, val in aggregated_output.items()}
        acc = aggregated_output['acc'] / aggregated_output['cnt']
        loss = aggregated_output['loss'] / aggregated_output['cnt']

        self.logger.experiment.add_scalar(f'Loss/{name}', loss, self.current_epoch)
        self.logger.experiment.add_scalar(f'Acc/{name}', acc, self.current_epoch)
        return acc, loss

    def validation_epoch_end(self, outputs):
        val_acc, val_loss = self._epoch_end(outputs, 'val_loss', 'val_acc', 'Val')
        self.log('val_acc', val_acc, logger=False, sync_dist=True)
        self.log('val_loss', val_loss, logger=False, sync_dist=True)

    def training_epoch_end(self, outputs):
        _, _ = self._epoch_end(outputs, 'loss', 'acc', 'Train')

    def test_epoch_end(self, outputs):
        _, _ = self._epoch_end(outputs, 'test_loss', 'test_acc', 'Test')


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
        self.num_workers = os.cpu_count() // 2

    def prepare_data(self):
        self.tokenizer_class.from_pretrained(pretrained_weights)

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        reviews_dataset = ReviewsDataset(df[0], tokenizer, df[1])
        train_size, val_size = int(self.train_perc * len(reviews_dataset)), int(self.val_perc * len(reviews_dataset))
        self.train_data, self.valid_data, self.test_data = random_split(reviews_dataset, [train_size, val_size, len(reviews_dataset) - train_size - val_size])

    def train_dataloader(self):
        # num_workers tips
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#num-workers
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=self.num_workers)


def version_from_params(p, shortcuts):
    return '__'.join([f'{short}_{p[orig]}' for orig, short in shortcuts.items()])


def get_best_ckp_acc(config_dir, metric='val_acc'):
    # given config dir, view all time directories and find the best checkpoint according to the metric
    best_score, best_ckp = 0, ''
    for d in os.listdir(config_dir):
        full_path = os.path.join(config_dir, d)
        if not os.path.isdir(full_path):
            continue

        ckp_path = os.path.join(full_path, 'checkpoints')
        if not os.path.exists(ckp_path):
            continue

        for ckp in os.listdir(ckp_path):
            if metric not in ckp:
                continue
            result = ckp.split('.')[0]
            result = result.split('--')[-1]
            result = float(result.split('=')[-1])
            if result >= best_score:
                best_ckp = os.path.join(ckp_path, ckp)
                best_score = result
    ic(best_ckp)
    return best_ckp


# %%
if __name__ == '__main__':
    # %%
    ic('start')

    warnings.filterwarnings('ignore')
    params = dict(
        train_size=0.8,
        val_size=0.1,
        seed=0xDEAD,
        batch=220,
        hidden=256,
        do=0.5,
        lr=3e-5,
        epochs=40,
        clip=1,
        save_fname='best_bert_sentiment.pt',
    )
    seed_everything(params['seed'], workers=True)

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

    # everything about profilers
    # https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#use-tensorboard-to-view-results-and-analyze-model-performance
    profiler = PyTorchProfiler(filename='pytorch_profiler')

    # when precision=16 Lightning uses native AMP ... automatic mixed precision
    trainer = pl.Trainer(gpus=2, strategy='ddp', logger=logger, profiler=profiler, num_sanity_val_steps=0, max_epochs=3, callbacks=cb, deterministic=True, precision=16)

    trainer.fit(pl_model, dataset)
    trainer.test(pl_model, dataset)

    # .......................................................... Loading from checkpoint .....................................................
    # hyppar_path = os.path.join(logs_dir, experiment_name, version_from_params(params, shortcuts))
    # if not os.path.isdir(hyppar_path):
    #     ic('training')
    #     trainer.fit(pl_model, dataset)
    #     trainer.test(pl_model, dataset)
    # else:
    #     ic('testing only')
    #     ckp = get_best_ckp_acc(hyppar_path)
    #     trainer.test(pl_model, dataset, ckpt_path=ckp)

    ic('end')
