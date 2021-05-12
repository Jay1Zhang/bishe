#!/usr/bin/env python
# -*- coding:utf-8 -*-

import glob
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils import *


# get frame number: filename[st:ed] from the filename string
def get_frame_no(filename, form='npy'):
    # return filename[filename.rfind('/') + 1: filename.rfind(f'.{form}')]  # Linux
    return filename[filename.rfind('\\') + 1 : filename.rfind(f'.{form}')]  # Win10


def load_segs(fold, filename, mode):
    full_filename = f'{FOLDS_DIR}/{mode}/{fold}/{filename}.data'
    segs = pickle.load(open(full_filename, 'rb'))
    return segs


# generate dataloader with qpsDataset
def gen_dataloader(fold, MODS):
    meta = pd.read_csv(META_FILE, index_col='seg_id')

    tra_seg_file = '5_2'    # key for loading the training segments
    val_seg_file = '5'      # key for loading the validation segments
    tes_seg_file = '5'      # key for loading testing segments

    tra_data = qpsDataset(MODS, meta, load_segs(fold, tra_seg_file, 'train'))
    val_data = qpsDataset(MODS, meta, load_segs(fold, val_seg_file, 'val'))
    tes_data = qpsDataset(MODS, meta, load_segs(fold, tes_seg_file, 'tes'))

    tra_loader = DataLoader(tra_data, batch_size=BATCH, shuffle=True, num_workers=N_WORKERS, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=N_WORKERS, drop_last=False)
    tes_loader = DataLoader(tes_data, batch_size=BATCH, shuffle=False, num_workers=N_WORKERS, drop_last=False)

    return tra_loader, val_loader, tes_loader


# qps dataset class
class qpsDataset(Dataset):
    def __init__(self, mods, meta, segs):
        self.mods = mods
        self.meta = meta
        self.segs = segs

        self.len = len(self.segs)       # length of dataset
        max_n_seg = max([len(seg) for seg in segs])  # max_n_seg = 5
        self.max_feat_len = {'a': 44 * max_n_seg, 'v': 70 * max_n_seg, 'l': 122 * max_n_seg}  # {a: 220, v: 350, l: 610}

        self.feat_src = './qps_dataset/'
        self.loadFeats = {'a': self.load_audio_feat, 'v': self.load_video_feat, 'l': self.load_lang_feat}

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ed_vote, change, dur_sec = self.meta.loc[self.segs[index][-1], ['ed_vote', 'change', 'dur_sec']]
        sample = {'ed_vote': ed_vote, 'change': change, 'dur': dur_sec}

        for mod in self.mods:
            data_list, msk_list = [], []
            used_segs = self.segs[index]
            for seg_id in used_segs:
                x, y = self.loadFeats[mod](f'{seg_id:04d}')
                data_list.append(x)
                msk_list.append(y)
            data = torch.cat(data_list, 0)
            msk = torch.cat(msk_list, 0)

            seq_len = data.size()[0]
            padding_size = list(data.size())
            padding_size[0] = self.max_feat_len[mod]
            # padding size: (S, E) - {a:[220, 73], v:[350, 512], l:[610, 200]}
            padding_data = torch.zeros(padding_size, dtype=torch.float)     # 零填充data
            padding_data[:seq_len] = data[:seq_len]
            padding_msk = torch.ones(padding_size[0], dtype=torch.bool)     # True填充mask
            padding_msk[:seq_len] = msk[:seq_len]

            sample[f'{mod}_data'] = padding_data    # (S, E)
            sample[f'{mod}_msk'] = padding_msk      # (S)

        return sample

# load acoustic features from covarep_norm.npy
    def load_audio_feat(self, seg):
        feat = np.load(f'{self.feat_src}/{seg}/covarep_norm.npy')
        return torch.from_numpy(feat), torch.zeros([feat.shape[0]], dtype=torch.bool)

    # load video features from frames in vgg_1fc/
    def load_video_feat(self, seg):
        form = 'npy'
        imgs, msk = [], []
        filenames = np.sort(np.array(glob.glob(f'{self.feat_src}/{seg}/vgg_1fc/*{form}')))

        min_frame = int(get_frame_no(filenames[0], form))
        max_frame = int(get_frame_no(filenames[-1], form))

        for frame in range(min_frame, max_frame+1):
            filename = f'{self.feat_src}/{seg}/vgg_1fc/{frame:05}.{form}'
            # 由于仅捕捉speaker的人脸，因而帧可能是不连续的，当某一帧不存在时需要补0
            if os.path.isfile(filename):
                feat = np.load(filename)
                msk.append(False)
            else:
                feat = np.zeros(512)
                msk.append(True)
            imgs.append(torch.from_numpy(feat).unsqueeze(0))    # (512,) -> (1, 512)

        return torch.cat(imgs, 0), torch.tensor(msk, dtype=torch.bool)

    # load language features from tencent_emb.npy
    def load_lang_feat(self, seg):
        feat = np.load(f'{self.feat_src}/{seg}/tencent_emb.npy').astype(np.float32)
        return torch.from_numpy(feat), torch.zeros([feat.shape[0]], dtype=torch.bool)


