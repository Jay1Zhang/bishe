#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import time
from torch import optim

import model
from utils import *

from dataset import gen_dataloader
from train import train_m2p2, eval_m2p2

if __name__ == '__main__':
    # 0 - Configure arguments parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--fd', required=False, default=9, type=int, help='fold id')
    parser.add_argument('--mod', required=False, default='avl', type=str,
                        help='modalities: a,v,l, or any combination of them')
    parser.add_argument('--dp', required=False, default=0.2, type=float, help='dropout')

    ## boolean flags
    parser.add_argument('--test_mode', default=False, action='store_true',
                        help='test mode: loading a pre-trained model and calculate loss')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='print more information')

    args = parser.parse_args()
    FOLD = int(args.fd)     # fold id
    MODS = list(args.mod)   # modalities: a, v, l
    DP = args.dp

    TEST_MODE = args.test_mode
    VERBOSE = args.verbose
    #TEST_MODE = True
    VERBOSE = True

    #############

    # 0 - Device configuration
    print("FOLD", FOLD)
    config_device()

    # 1 - load dataset
    tra_loader, val_loader, tes_loader = gen_dataloader(FOLD, MODS)

    # 2 - Initialize m2p2 models
    # initialize multiple models to output the latent embeddings for a,v,l
    latent_models = {mod: model.LatentModule(mod, DP).to(device) for mod in MODS}
    # initialize align models
    align_model = model.AlignModule().to(device)
    # initialize pers model
    pers_model = model.PersModel(nmod=len(MODS), nfeat=N_FEATS, dropout=DP).to(device)

    # initialize m2p2 models and hyper-parameters and optimizer
    m2p2_models = latent_models
    m2p2_models['align'] = align_model
    #m2p2_models['het'] = het_model
    m2p2_models['pers'] = pers_model

    m2p2_params = get_hyper_params(m2p2_models)
    m2p2_optim = optim.Adam(m2p2_params, lr=LR, weight_decay=W_DECAY)
    m2p2_scheduler = optim.lr_scheduler.StepLR(m2p2_optim, step_size=STEP_SIZE, gamma=SCHE_GAMMA)

    #if VERBOSE:
    #   print('####### total m2p2 hyper-parameters ', count_hyper_params(m2p2_params))
    #   for k, v in m2p2_models.items():
    #       print(v)
    #       print(count_hyper_params(v.parameters()))

    # 3 - Initialize reference models and weights: w_a, w_v, w_l
    weight_mod = {mod: 1. / len(MODS) for mod in MODS}
    # initialize het models
    het_models = {'het': model.HetModule(MODS).to(device)}
    het_params = get_hyper_params(het_models)
    het_optim = optim.Adam(het_params, lr=LR, weight_decay=W_DECAY)

    # 4 - Train or Test
    if not TEST_MODE:
        min_loss_pers = 1e5
        #### Master Procedure Start ####
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            # train m2p2 model
            train_loss_pers, train_loss_ref = train_m2p2(MODS, tra_loader, m2p2_models, m2p2_optim, m2p2_scheduler,
                                                                    het_models, het_optim, weight_mod)
            # eval and save m2p2 model
            eval_loss_pers, eval_loss_ref_mod = eval_m2p2(MODS, val_loader, m2p2_models, het_models, weight_mod)
            # update weight_mod
            weight_mod = update_weight_mod(MODS, old_weight_mod=weight_mod, loss_ref_mod=eval_loss_ref_mod)
            # save m2p2 model
            if eval_loss_pers < min_loss_pers:
                print(f'[SAVE MODEL] eval pers loss: {eval_loss_pers:.5f}\tmini pers loss: {min_loss_pers:.5f}')
                min_loss_pers = eval_loss_pers
                saveModel(FOLD, m2p2_models, weight_mod)

            # output loss information
            end_time = time.time()
            if VERBOSE:
                epoch_mins, epoch_secs = calc_epoch_time(start_time, end_time)
                print(f'Epoch: {epoch + 1:02}/{N_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain persuasion loss:{train_loss_pers:.5f}\tTrain reference Loss:{train_loss_ref:.5f}')
                print(f'\tEval persuasion loss:{eval_loss_pers:.5f}')
        #### Master Procedure End ####

    else:
        m2p2_models, weight_mod = loadModel(FOLD, m2p2_models)
        test_loss_pers, _ = eval_m2p2(MODS, tes_loader, m2p2_models, het_models, weight_mod)
        print(f'Test persuasion loss:{test_loss_pers:.5f}')
        print('MSE:', round(test_loss_pers, 3))
