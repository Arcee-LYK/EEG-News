from model import DCmodel2
from functools import partial
from torch import nn
import numpy as np
import os
from tqdm import tqdm
import torch
import warnings
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import chi2_contingency, fisher_exact
import argparse
from torch.utils.data import DataLoader
from data import all_data, high_low_data, ai_human_data, Dataset
from copy import deepcopy
warnings.filterwarnings('ignore')

def experiment(train_list, eval_list, args, device, model_file):
    # eeg_test_dataset = all_data(train_list, eval_list, feature=args.feature, type='test')
    eeg_test_dataset = ai_human_data(train_list, eval_list, involve=args.involve, feature=args.feature, type='test')
    # eeg_test_dataset = high_low_data(train_list, eval_list, person=args.person, feature=args.feature, type='test')

    test_loader = DataLoader(eeg_test_dataset, batch_size=len(eeg_test_dataset), shuffle=False)

    model = DCmodel2(embed_dim=32, num_classes=2, eeg_seq_len=5, eeg_dim=160, depth=3,
                 num_heads=4, qkv_bias=True, mixffn_start_layer_index=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['state_dict'])
    

    # for epoch in tqdm(range(epochs)):
    acc, f1, p_value = eval(model, test_loader)

    return acc, f1, p_value

def eval(model, eval_loader):
    model.eval()
    with torch.no_grad():
        predicted = []
        labels = []
        for eeg, y in eval_loader:
            eeg = eeg.float().cuda()
            y = y.float()
            pred = model(eeg=eeg)
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
            predicted.append(pred)
            labels.append(y)
    accuracy = accuracy_score(np.concatenate(labels), np.concatenate(predicted))
    f1 = f1_score(np.concatenate(labels), np.concatenate(predicted),average='weighted')
    labels = np.concatenate(labels)
    predicted = np.concatenate(predicted)
    conf_matrix = np.zeros((2, 2), dtype=int)
    classes = [0, 1] 
    for true, pred in zip(labels, predicted):
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)
        conf_matrix[true_idx, pred_idx] += 1
    if (conf_matrix == 0).any():
        print("检测到 0 频数，使用 Fisher’s Exact Test 替代卡方检验")
        odds_ratio, p_value = fisher_exact(conf_matrix)
        # p_value = 1
    else:
        chi2_stat, p_value, _, _ = chi2_contingency(conf_matrix)
    print(p_value)
    return accuracy, f1, p_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='de', help='de or psd', choices=['de', 'psd'])
    parser.add_argument('--involve', type=str, default='high', help='high or low', choices=['high', 'low'])
    parser.add_argument('--person', type=str, default='human', help='human or ai', choices=['human', 'ai'])
    # parser.add_argument('--eval_folder', type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    device = torch.device("cuda:0")
    
    file_list = [i for i in range(1, 9)]
    eval_lists = [[1, 2], [3, 4], [5, 6], [7, 8]]
    ACC = []
    F1 = []
    P = []
    for i in range(1, 5):
        eval_list = eval_lists[i-1]
        train_list = deepcopy(file_list)
        train_list.remove(eval_list[0])
        train_list.remove(eval_list[1])
        # model_file = f'all_dc/{args.feature}/{i}/model.pt'
        model_file = f'human_ai_dc_nocrossatten/{args.involve}/{args.feature}/{i}/model.pt'
        acc, f1, p_value = experiment(train_list, eval_list, args, device, model_file)
        # print(acc)
        ACC.append(acc)
        F1.append(f1)
        P.append(p_value)
    print("-----------")
    print(np.mean(ACC), np.std(ACC))
    print(np.mean(F1), np.std(F1))
    print(np.mean(P))