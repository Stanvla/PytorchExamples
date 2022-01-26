import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Sampler, Dataset


def collate_fn(batch):
    inputs = [x['tokenized'] for x in batch]
    labels = [x['label']for x in batch]

    # pad sequences for the max length and create mask
    max_len = max(s.shape[0] for s in inputs)
    padded_seqs, masks = [], []

    for s in inputs:
        padded_seq = F.pad(s, [0, max_len - len(s)])
        padded_seqs.append(padded_seq)

        mask = F.pad(torch.ones(len(s)), [0, max_len - len(s)])
        masks.append(mask)

    return {"inputs": torch.stack(padded_seqs), "labels": torch.FloatTensor(labels), 'attention_mask' : torch.stack(masks)}


class ReviewsDataset(Dataset):
    def __init__(self, reviews, tokenizer, labels):
        self.labels = labels
        # tokenized reviews
        self.tokenized = [tokenizer.encode(review) for review in reviews]

    def __getitem__(self, idx):
        return {"tokenized": torch.tensor(self.tokenized[idx]), "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


class ReviewsSampler(Sampler):
    def __init__(self, subset, batch_size):
        self.batch_size = batch_size
        self.subset = subset

        self.indices = subset.indices
        # tokenized for our data
        self.tokenized = np.array(subset.dataset.tokenized)[self.indices]

    def __iter__(self):
        batch_idx = []
        # index in sorted data
        for index in np.argsort(list(map(len, self.tokenized))):
            batch_idx.append(index)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []

        if len(batch_idx) > 0:
            yield batch_idx

    def __len__(self):
        return len(self.subset.dataset)