import sys
sys.path.append('..')
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
import numpy as np

from data import EncodedDataset, MapDataset, SKLearnEncoder, SurvDataset
from fastai.data.load import DataLoader as FastaiTfmdDL


__all__ = ['BagDataset', 'make_dataset', 'get_cohort_df']


@dataclass
class BagDataset(Dataset):
    """A dataset of bags of instances."""
    bags: Sequence[Iterable[Path]]
    """The `.h5` files containing the bags.

    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x
    F, where N is the number of instances and F the number of features per
    instance.
    """
    bag_size: Optional[int] = None
    """The number of instances in each bag.

    For bags containing more instances, a random sample of `bag_size`
    instances will be drawn.  Smaller bags are padded with zeros.  If
    `bag_size` is None, all the samples will be used.
    """

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # collect all the features
        feats = []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, 'r') as f:
                feats.append(torch.from_numpy(f['feats'][:]))
        feats = torch.concat(feats).float()

        # sample a subset, if required
        if self.bag_size:
            return _to_fixed_size_bag(feats, bag_size=self.bag_size)
        else:
            return feats, len(feats)


def _to_fixed_size_bag(bag: torch.Tensor, bag_size: int = 512) -> Tuple[torch.Tensor, int]:
    # get up to bag_size elements
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]

    # zero-pad if we don't have enough samples
    zero_padded = torch.cat((bag_samples,
                             torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1])))
    return zero_padded, min(bag_size, len(bag))


def make_dataset(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, Sequence[Any]],
    add_features: Optional[Iterable[Tuple[Any, Sequence[Any]]]] = None,
    bag_size: Optional[int] = None,
) -> MapDataset:
    if add_features:
        return _make_multi_input_dataset(
            bags=bags, targets=targets, add_features=add_features, bag_size=bag_size)
    else:
        return _make_basic_dataset(
            bags=bags, target_enc=targets[0], targs=targets, bag_size=bag_size)


def get_target_enc(mil_learn):
    return mil_learn.dls.train.dataset._datasets[-1].encode


def _make_basic_dataset(
    *,
    bags: Sequence[Iterable[Path]],
    target_enc: SKLearnEncoder,
    targs: Sequence[Any],
    bag_size: Optional[int] = None,
) -> MapDataset:
    assert len(bags) == len(targs), \
        'number of bags and ground truths does not match!'

    ds = MapDataset(
        zip_bag_targ,
        BagDataset(bags, bag_size=bag_size),
        # EncodedDataset(target_enc, targs),
        SurvDataset(targs)
    )

    return ds


def zip_bag_targ(bag, targets):
    features, lengths = bag
    return (
        features,
        lengths,
        targets.squeeze(),
    )


def _make_multi_input_dataset(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, Sequence[Any]],
    add_features: Iterable[Tuple[Any, Sequence[Any]]],
    bag_size: Optional[int] = None
) -> MapDataset:
    target_enc, targs = targets
    assert len(bags) == len(targs), \
        'number of bags and ground truths does not match!'
    for i, (_, vals) in enumerate(add_features):
        assert len(vals) == len(targs), \
            f'number of additional attributes #{i} and ground truths does not match!'

    bag_ds = BagDataset(bags, bag_size=bag_size)

    add_ds = MapDataset(
        _splatter_concat,
        *[
            EncodedDataset(enc, vals)
            for enc, vals in add_features
        ])

    targ_ds = EncodedDataset(target_enc, targs)

    ds = MapDataset(
        _attach_add_to_bag_and_zip_with_targ,
        bag_ds,
        add_ds,
        targ_ds,
    )

    return ds


def _splatter_concat(*x): return torch.concat(x, dim=1)


def _attach_add_to_bag_and_zip_with_targ(bag, add, targ):
    return (
        torch.concat([
            bag[0],  # the bag's features
            add.repeat(bag[0].shape[0], 1)  # the additional features
        ], dim=1),
        bag[1],  # the bag's length
        targ.squeeze(),   # the ground truth
    )


def get_cohort_df(
    clini_excel: Path, slide_csv: Path, feature_dir: str,
    target_label: str, categories: Iterable[str]
) -> pd.DataFrame:
    suffix = Path(clini_excel).suffix
    if suffix == '.csv':
        clini_df = pd.read_csv(clini_excel)
    else:
        clini_df = pd.read_excel(clini_excel)
    slide_df = pd.read_csv(slide_csv)

    df = clini_df.merge(slide_df, on='PATIENT')

    # remove uninteresting
    # df = df[df[target_label].isin(categories)]
    # remove slides we don't have
    slides = set(feature_dir.glob('*.h5'))
    slide_df = pd.DataFrame(slides, columns=['slide_path'])
    slide_df['FILENAME'] = slide_df.slide_path.map(lambda p: p.stem)
    df['FILENAME'] = df['FILENAME'].astype('string')

    # df = df.drop(columns='slide_path')

    df = df.merge(slide_df, on='FILENAME')

    # reduce to one row per patient with list of slides in `df['slide_path']`
    patient_df = df.groupby('PATIENT').first().drop(columns='slide_path')
    patient_slides = df.groupby('PATIENT').slide_path.apply(set).apply(list)
    df = patient_df.merge(patient_slides, left_on='PATIENT',
                          right_index=True).reset_index()

    return df


class EPBSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        assert batch_size <= len(data_source), "Batch size cannot be larger than the size of the dataset"

        self.total_indices = list(range(len(data_source)))
        np.random.shuffle(self.total_indices)
        #print('total_indices:', self.total_indices)
        self.event_indices = [i for i, data in enumerate(data_source) if data[2][1] > 0.1]  # hard-coding, need improvement
        np.random.shuffle(self.event_indices)
        #print('event_indices:', self.event_indices)
        self.event_iter = iter(self.event_indices)

    def __iter__(self):
            #total_indices = list(range(len(self.data_source)))
            #np.random.shuffle(total_indices)
            #event_iter = iter(np.random.permutation(self.event_indices))
            batch = []
            for idx in self.total_indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    #print('original batch:', batch)
                    if self.event_indices:
                        try:
                            replace_inx = next(self.event_iter)
                            #print('replace_inx:', replace_inx)
                            if replace_inx not in batch:
                                batch[np.random.randint(len(batch))] = replace_inx
                        except StopIteration:
                            self.event_iter = iter(np.random.permutation(self.event_indices))
                            #batch[np.random.randint(len(batch))] = next(self.event_iter)
                            replace_idx = next(self.event_iter)
                            if replace_idx not in batch:
                                batch[np.random.randint(len(batch))] = replace_idx
                    #print(f"Yielding batch: {batch}")  
                    yield batch
                    #print(f"Yielding batch of size: {len(batch)}")  
                    #print(f"Yielding batch: {batch}")
                    batch = []
            if batch and len(batch) > 1 and not self.drop_last:
                if self.event_indices:
                    try:
                        batch[np.random.randint(len(batch))] = next(self.event_iter)
                    except StopIteration:
                        self.event_iter = iter(np.random.permutation(self.event_indices))
                        batch[np.random.randint(len(batch))] = next(self.event_iter)
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.total_indices) // self.batch_size
        else:
            return (len(self.total_indices) + self.batch_size - 1) // self.batch_size


def epb_collate_fn(batch):
    #print(f"Received batch: {batch}")
    batch_feats, batch_lens, batch_targets = zip(*batch)
    batch_feats = torch.stack(batch_feats)
    batch_lens = torch.tensor(batch_lens)
    batch_targets = torch.stack(batch_targets)
    #print(f"Collated batch_feats: {batch_feats.shape}, batch_lens: {batch_lens.shape}, batch_targets: {batch_targets.shape}")
    return batch_feats, batch_lens, batch_targets



class EPBDataLoader(FastaiTfmdDL):
    def __init__(self, dataset, batch_sampler, collate_func, **kwargs):
        self.batch_sampler = batch_sampler
        self.collate_func = collate_func
        self.sampler_iter = iter(batch_sampler)
        super().__init__(dataset, **kwargs)

    def __iter__(self):
        for batch in self.batch_sampler:
            yield self.collate_func([self.dataset[i] for i in batch])

    def get_idxs(self):
        return next(self.sampler_iter)


    def create_batches(self, samps):
        return super().create_batches(samps)

    def do_batch(self, b):
        """rewrite do_batch to use custom_collate_fn ."""
        batch = self.collate_func(b)
        return self.retain(batch, b)  # retain to ensure data type