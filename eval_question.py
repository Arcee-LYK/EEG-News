from model import DCmodel2
from functools import partial
from torch import nn
import numpy as np
import os
from tqdm import tqdm
import torch
import warnings
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.stats import chi2_contingency, fisher_exact
import argparse
from torch.utils.data import DataLoader
from data import data_for_question_list
from copy import deepcopy
warnings.filterwarnings('ignore')

def experiment(file_path, file_list, args, device, model_file):
    eeg_test_dataset = data_for_question_list(file_path=file_path, file_list=file_list, feature=args.feature,
                                          question=args.question, eval_video_ind=args.eval_video_ind, type='test')

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
    accuracy = balanced_accuracy_score(np.concatenate(labels), np.concatenate(predicted))
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
    parser.add_argument('--data_folder', type=str, default='ai_high',choices=['ai_high', 'ai_low', 'human_high', 'human_low'])
    parser.add_argument('--feature', type=str, default='de', help='de or psd', choices=['de', 'psd'])
    parser.add_argument('--question', type=int, default=1)
    # parser.add_argument('--eval_video_ind', type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    device = torch.device("cuda:0")
    
    file_list = [i for i in range(1, 9)]
    file_path = f'../data/new_data/mat_features_four_class/{args.data_folder}'
    ACC = []
    F1 = []
    P = []
    for i in range(1, 5):
        args.eval_video_ind = i
        model_file = f'question_results_nocrossatten/{args.data_folder}/{args.feature}/question{args.question}_videoind{i}_model.pt'
        acc, f1, p_value = experiment(file_path, file_list, args, device, model_file)
        # print(acc)
        ACC.append(acc)
        F1.append(f1)
        P.append(p_value)
    print("-----------")
    print(np.mean(ACC), np.std(ACC))
    print(np.mean(F1), np.std(F1))
    print(np.mean(P))

    #cross attention                                              no cross attention
    #                 de                 psd                    de                  psd
    #ai_high      62.82/5.30*          58.88/2.86*            60.65/3.42          62.76/4.69
    #             68.71/7.50           63.46/7.91             65.70/8.01          68.00/5.08
    #ai_low       59.46/1.87*          65.03/2.11*            57.99/2.53          59.76/3.44
    #             59.09/3.04           65.37/2.42             56.17/3.88          58.86/4.50
    #human_high   66.96/1.70*          61.82/2.20*            62.80/4.84          63.14/4.43
    #             73.27/4.01           72.02/8.05             68.05/8.39          67.45/9.53
    #human_low    68.35/5.79*          69.89/5.00*            69.22/5.21          70.80/4.09
    #             69.65/3.72           68.22/6.96             69.29/4.91          68.98/4.45