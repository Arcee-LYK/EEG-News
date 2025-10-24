from model import DCmodel2
from functools import partial
from torch import nn
import numpy as np
import os
from tqdm import tqdm
import torch
from training_question import MLPTrainer
import warnings
import argparse
from torch.utils.data import DataLoader
from data import data_for_question_list
warnings.filterwarnings('ignore')

def experiment(lr, weight_decay, file_path, file_list, args, device):
    batch_size = 10240
    epochs = 200

    eeg_train_dataset = data_for_question_list(file_path=file_path, file_list=file_list, feature=args.feature, 
                                          question=args.question, eval_video_ind=args.eval_video_ind, type='train')
    eeg_test_dataset = data_for_question_list(file_path=file_path, file_list=file_list, feature=args.feature,
                                          question=args.question, eval_video_ind=args.eval_video_ind, type='test')


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
    parser.add_argument('--data_folder', type=str, default='ai_high',choices=['ai_high', 'ai_low', 'human_high', 'human_low'])
    parser.add_argument('--feature', type=str, default='de', help='de or psd', choices=['de', 'psd'])
    parser.add_argument('--question', type=int, default=1)
    parser.add_argument('--eval_video_ind', type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    device = torch.device("cuda:0")
    max_accuracy = 0
    max_model = None
    file_list = [i for i in range(1, 9)]
    file_path = f'../data/new_data/mat_features_four_class/{args.data_folder}'
    for lr in [1e-3, 1e-4, 1e-5]:
        for weight_decay in [1e-2, 1e-4, 1e-6]:
            model, acc = experiment(lr=lr, weight_decay=weight_decay,
                                    file_path=file_path, file_list=file_list, args=args, device=device)
            if acc > max_accuracy:
                max_accuracy = acc
                max_model = model
    directory = "question_results_nocrossatten/{}/{}".format(args.data_folder, args.feature)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({'state_dict': max_model}, directory + f'/question{args.question}_videoind{args.eval_video_ind}_model.pt')
    print("max acc:", max_accuracy)
