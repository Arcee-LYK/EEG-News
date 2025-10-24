from model import DCmodel2
from functools import partial
from torch import nn
import numpy as np
import os
from tqdm import tqdm
import torch
from training import MLPTrainer
import warnings
import argparse
from torch.utils.data import DataLoader
from data import ai_human_data, Dataset
warnings.filterwarnings('ignore')

def experiment(lr, weight_decay, train_list, eval_list, args, device):
    batch_size = 10240
    epochs = 200

    eeg_train_dataset = ai_human_data(train_list, eval_list, involve=args.involve, feature=args.feature, type='train')
    eeg_test_dataset = ai_human_data(train_list, eval_list, involve=args.involve, feature=args.feature, type='test')


    train_loader = DataLoader(eeg_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(eeg_test_dataset, batch_size=batch_size, shuffle=False)
    # model = DCmodel(eeg_dim=160, embed_dim=32, d_model=32, n_embeddings=400, class_num=2).to(device)
    model = DCmodel2(embed_dim=32, num_classes=2, eeg_seq_len=5, eeg_dim=160, depth=3,
                 num_heads=4, qkv_bias=True, mixffn_start_layer_index=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))

    mlp_trainer = MLPTrainer(model, optimizer, scheduler, device, print_freq=200)

    # for epoch in tqdm(range(epochs)):
    mlp_trainer.train(train_loader, test_loader, epochs)

    return mlp_trainer.max_model, mlp_trainer.max_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--involve', type=str, default='high', help='high or low', choices=['high', 'low'])
    parser.add_argument('--feature', type=str, default='de', help='de or psd', choices=['de', 'psd'])
    parser.add_argument('--eval_folder', type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    device = torch.device("cuda:0")
    max_accuracy = 0
    max_model = None
    train_list = [i for i in range(1, 9)]
    train_list.remove(args.eval_folder * 2 - 1)
    train_list.remove(args.eval_folder * 2)
    eval_list = [args.eval_folder * 2 - 1, args.eval_folder * 2]
    for lr in [1e-3, 1e-4, 1e-5]:
        for weight_decay in [1e-2, 1e-4, 1e-6]:
            model, acc = experiment(lr=lr, weight_decay=weight_decay,
                                    train_list=train_list, eval_list=eval_list, args=args, device=device)
            if acc > max_accuracy:
                max_accuracy = acc
                max_model = model
    directory = "human_ai_dc_nocrossatten/{}/{}/{}".format(args.involve, args.feature, args.eval_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({'state_dict': max_model}, directory + '/model.pt')
    print("max acc:", max_accuracy)
