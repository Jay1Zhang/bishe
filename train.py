#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch

from utils import *


def gen_align_emb(bi_attn_emb, tri_attn_emb):
    # generate H^align with H^s_m
    # align_emb = torch.cat([bi_attn_emb, tri_attn_emb], dim=2)    # 在特征维度上拼接
    align_emb = bi_attn_emb
    align_emb = torch.mean(align_emb, dim=1)    # 在序列维度上取平均
    return align_emb


def gen_het_emb(MODS, latent_emb_mod, weight_mod):
    # generate H^het with H^latent_m, w_m
    het_emb_mod = {}
    for mod in MODS:
        het_emb_mod[mod] = torch.tensor(weight_mod[mod]) * latent_emb_mod[mod]
    # het_emb_mod = [torch.tensor(weight_mod[mod]) * latent_emb_mod[mod] for mod in MODS]
    het_emb = torch.cat([v for v in het_emb_mod.values()], dim=1)   # 在序列维度上拼接
    het_emb = torch.max(het_emb, dim=1)[0]  # 在序列维度上取max, 更加符合前期推理
    return het_emb


def gen_meta_emb(sample):
    st_vote = (sample['ed_vote'] - sample['change']).float().unsqueeze(1).to(device)
    dur_time = (sample['dur'] / MAX_DUR).float().unsqueeze(1).to(device)
    meta_emb = torch.cat([st_vote, dur_time], dim=1)
    return meta_emb


def inference(MODS, sample_batched, m2p2_models, het_models, weight_mod):
    latent_emb_mod = {}
    for mod in MODS:
        latent_emb_mod[mod] = m2p2_models[mod](sample_batched[f'{mod}_data'].to(device))

    bi_attn_emb, tri_attn_emb = m2p2_models['align'](latent_emb_mod)
    align_emb = gen_align_emb(bi_attn_emb, tri_attn_emb)
    het_emb = gen_het_emb(MODS, latent_emb_mod, weight_mod)
    meta_emb = gen_meta_emb(sample_batched)

    y_pred_ref_mod = {mod: het_models[mod](latent_emb_mod[mod]) for mod in MODS}
    y_pred = m2p2_models['pers'](align_emb, het_emb, meta_emb)   # (N, 1)
    y_true = sample_batched['ed_vote'].float().to(device)   # (N)

    # calc loss
    loss_pers = calcPersLoss(y_pred, y_true)
    loss_ref_mod = {mod: calcPersLoss(y_pred_ref_mod[mod], y_true) for mod in MODS}
    # loss_ref = sum(list(loss_ref_mod.values()))
    loss_ref_weighted = sum([torch.tensor(weight_mod[mod]) * loss_ref_mod[mod] for mod in MODS])

    return loss_pers, loss_ref_mod, loss_ref_weighted


def train_m2p2(MODS, iterator, m2p2_models, m2p2_optim, m2p2_scheduler,
                          het_models, het_optim, weight_mod):
    setModelMode(m2p2_models, is_train_mode=True)
    setModelMode(het_models, is_train_mode=True)
    total_loss_pers = 0
    total_loss_ref = 0

    for i_batch, sample_batched in enumerate(iterator):
        m2p2_optim.zero_grad()
        het_optim.zero_grad()
        # forward
        loss_pers, loss_ref_mod, loss_ref_weighted = \
            inference(MODS, sample_batched, m2p2_models, het_models, weight_mod)
        total_loss_pers += loss_pers.item()
        total_loss_ref += loss_ref_weighted.item()

        # backward
        loss_ref_weighted.backward(retain_graph=True)
        het_optim.step()
        loss_pers.backward()   # 防止缓冲区被释放
        m2p2_optim.step()

    m2p2_scheduler.step()
    aver_loss_pers = total_loss_pers / (i_batch + 1)
    aver_loss_ref = total_loss_ref / (i_batch + 1)

    return aver_loss_pers, aver_loss_ref


def eval_m2p2(MODS, iterator, m2p2_models, het_models, weight_mod):
    setModelMode(m2p2_models, is_train_mode=False)
    setModelMode(het_models, is_train_mode=False)

    total_loss_pers = 0
    total_loss_ref_mod = {mod: 0 for mod in MODS}

    for i_batch, sample_batched in enumerate(iterator):
        # forward
        with torch.no_grad():
            loss_pers, loss_ref_mod, loss_ref_weighted = \
                inference(MODS, sample_batched, m2p2_models, het_models, weight_mod)
            total_loss_pers += loss_pers.item()
            for mod in MODS:
                total_loss_ref_mod[mod] += loss_ref_mod[mod].item()

    aver_loss_pers = total_loss_pers / (i_batch + 1)
    aver_loss_ref_mod = {mod: total_loss_ref_mod[mod] / (i_batch + 1) for mod in MODS}

    return aver_loss_pers, aver_loss_ref_mod

