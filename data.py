import torch
import mne
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
from sklearn import preprocessing
from openpyxl import load_workbook
channels = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 
            'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']

specific_channels = ['O1', 'Oz', 'O2', 'F3', 'Fz', 'F4', 'T7', 'T8']


class ai_human_data():
    def __init__(self, train_list, eval_list, involve='high', feature='de', type='train'):
        if involve == 'high':
            file_list_human = '../data/new_data/mat_features_four_class/human_high'
            file_list_ai = '../data/new_data/mat_features_four_class/ai_high'
        else:
            file_list_human = '../data/new_data/mat_features_four_class/human_low'
            file_list_ai = '../data/new_data/mat_features_four_class/ai_low'
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for id in train_list:
            file_path_human = os.path.join(file_list_human, str(id) + '.mat')
            file_path_ai = os.path.join(file_list_ai, str(id) + '.mat')
            data_human = sio.loadmat(file_path_human)
            data_ai = sio.loadmat(file_path_ai)
            for i in range(1, 13):
                human_data = data_human[f'video_{feature}_feature_{i}']
                X_train.append(human_data)
                for j in range(human_data.shape[0]):
                    y_train.append(0)
                ai_data = data_ai[f'video_{feature}_feature_{i}']
                X_train.append(ai_data)
                for j in range(ai_data.shape[0]):
                    y_train.append(1)
        for id in eval_list:
            file_path_human = os.path.join(file_list_human, str(id) + '.mat')
            file_path_ai = os.path.join(file_list_ai, str(id) + '.mat')
            data_human = sio.loadmat(file_path_human)
            data_ai = sio.loadmat(file_path_ai)
            for i in range(1, 13):
                human_data = data_human[f'video_{feature}_feature_{i}']
                X_test.append(human_data)
                for j in range(human_data.shape[0]):
                    y_test.append(0)
                ai_data = data_ai[f'video_{feature}_feature_{i}']
                X_test.append(ai_data)
                for j in range(ai_data.shape[0]):
                    y_test.append(1)
        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        _, p, b = X_train.shape
        X_train = np.reshape(X_train, [X_train.shape[0], p*b])
        X_test = np.reshape(X_test, [X_test.shape[0], p*b])
        eeg_scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = eeg_scaler.transform(X_train)
        X_test = eeg_scaler.transform(X_test)

        if type == 'train':
            self.X = torch.from_numpy(X_train)
            self.y = torch.from_numpy(y_train)
            self.len = X_train.shape[0]
        else:
            self.X = torch.from_numpy(X_test)
            self.y = torch.from_numpy(y_test)
            self.len = X_test.shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.X[item], self.y[item]


class high_low_data():
    def __init__(self, train_list, eval_list, person='human', feature='de', type='train'):
        if person == 'human':
            file_list_high = '../data/new_data/mat_features_four_class/human_high'
            file_list_low = '../data/new_data/mat_features_four_class/human_low'
        else:
            file_list_high = '../data/new_data/mat_features_four_class/ai_high'
            file_list_low = '../data/new_data/mat_features_four_class/ai_low'
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for id in train_list:
            file_path_high = os.path.join(file_list_high, str(id) + '.mat')
            file_path_low = os.path.join(file_list_low, str(id) + '.mat')
            data_high = sio.loadmat(file_path_high)
            data_low = sio.loadmat(file_path_low)
            for i in range(1, 13):
                high_data = data_high[f'video_{feature}_feature_{i}']
                X_train.append(high_data)
                for j in range(high_data.shape[0]):
                    y_train.append(0)
                low_data = data_low[f'video_{feature}_feature_{i}']
                X_train.append(low_data)
                for j in range(low_data.shape[0]):
                    y_train.append(1)
        for id in eval_list:
            file_path_high = os.path.join(file_list_high, str(id) + '.mat')
            file_path_low = os.path.join(file_list_low, str(id) + '.mat')
            data_high = sio.loadmat(file_path_high)
            data_low = sio.loadmat(file_path_low)
            for i in range(1, 13):
                high_data = data_high[f'video_{feature}_feature_{i}']
                X_test.append(high_data)
                for j in range(high_data.shape[0]):
                    y_test.append(0)
                low_data = data_low[f'video_{feature}_feature_{i}']
                X_test.append(low_data)
                for j in range(low_data.shape[0]):
                    y_test.append(1)
        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        _, p, b = X_train.shape
        X_train = np.reshape(X_train, [X_train.shape[0], p * b])
        X_test = np.reshape(X_test, [X_test.shape[0], p * b])
        eeg_scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = eeg_scaler.transform(X_train)
        X_test = eeg_scaler.transform(X_test)

        if type == 'train':
            self.X = torch.from_numpy(X_train)
            self.y = torch.from_numpy(y_train)
            self.len = X_train.shape[0]
        else:
            self.X = torch.from_numpy(X_test)
            self.y = torch.from_numpy(y_test)
            self.len = X_test.shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.X[item], self.y[item]


class all_data():
    def __init__(self, train_list, eval_list, feature='de', type='train'):
        file_list_human_high = '../data/new_data/mat_features_four_class/human_high'
        file_list_human_low = '../data/new_data/mat_features_four_class/human_low'
        file_list_ai_high = '../data/new_data/mat_features_four_class/ai_high'
        file_list_ai_low = '../data/new_data/mat_features_four_class/ai_low'

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for id in train_list:
            file_path_human_high = os.path.join(file_list_human_high, str(id) + '.mat')
            file_path_human_low = os.path.join(file_list_human_low, str(id) + '.mat')
            file_path_ai_high = os.path.join(file_list_ai_high, str(id) + '.mat')
            file_path_ai_low = os.path.join(file_list_ai_low, str(id) + '.mat')
            data_human_high = sio.loadmat(file_path_human_high)
            data_human_low = sio.loadmat(file_path_human_low)
            data_ai_high = sio.loadmat(file_path_ai_high)
            data_ai_low = sio.loadmat(file_path_ai_low)
            for i in range(1, 13):
                human_high_data = data_human_high[f'video_{feature}_feature_{i}']
                X_train.append(human_high_data)
                for j in range(human_high_data.shape[0]):
                    y_train.append(0)
                human_low_data = data_human_low[f'video_{feature}_feature_{i}']
                X_train.append(human_low_data)
                for j in range(human_low_data.shape[0]):
                    y_train.append(1)
                ai_high_data = data_ai_high[f'video_{feature}_feature_{i}']
                X_train.append(ai_high_data)
                for j in range(ai_high_data.shape[0]):
                    y_train.append(2)
                ai_low_data = data_ai_low[f'video_{feature}_feature_{i}']
                X_train.append(ai_low_data)
                for j in range(ai_low_data.shape[0]):
                    y_train.append(3)
        for id in eval_list:
            file_path_human_high = os.path.join(file_list_human_high, str(id) + '.mat')
            file_path_human_low = os.path.join(file_list_human_low, str(id) + '.mat')
            file_path_ai_high = os.path.join(file_list_ai_high, str(id) + '.mat')
            file_path_ai_low = os.path.join(file_list_ai_low, str(id) + '.mat')
            data_human_high = sio.loadmat(file_path_human_high)
            data_human_low = sio.loadmat(file_path_human_low)
            data_ai_high = sio.loadmat(file_path_ai_high)
            data_ai_low = sio.loadmat(file_path_ai_low)
            for i in range(1, 13):
                human_high_data = data_human_high[f'video_{feature}_feature_{i}']
                X_test.append(human_high_data)
                for j in range(human_high_data.shape[0]):
                    y_test.append(0)
                human_low_data = data_human_low[f'video_{feature}_feature_{i}']
                X_test.append(human_low_data)
                for j in range(human_low_data.shape[0]):
                    y_test.append(1)
                ai_high_data = data_ai_high[f'video_{feature}_feature_{i}']
                X_test.append(ai_high_data)
                for j in range(ai_high_data.shape[0]):
                    y_test.append(2)
                ai_low_data = data_ai_low[f'video_{feature}_feature_{i}']
                X_test.append(ai_low_data)
                for j in range(ai_low_data.shape[0]):
                    y_test.append(3)
        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        _, p, b = X_train.shape
        X_train = np.reshape(X_train, [X_train.shape[0], p * b])
        X_test = np.reshape(X_test, [X_test.shape[0], p * b])
        eeg_scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = eeg_scaler.transform(X_train)
        X_test = eeg_scaler.transform(X_test)
        if type == 'train':
            self.X = torch.from_numpy(X_train)
            self.y = torch.from_numpy(y_train)
            self.len = X_train.shape[0]
        else:
            self.X = torch.from_numpy(X_test)
            self.y = torch.from_numpy(y_test)
            self.len = X_test.shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.X[item], self.y[item]

class Dataset():
    def __init__(self, eeg, eeg2, label, time_window=5):
        super().__init__()
        self.eeg = eeg
        self.eeg2 = eeg2
        self.label = label
        self.time_window = time_window

        self.dic = {}
        self.cnt = 0
        for i in range(label.shape[0] - time_window):
            if label[i] == label[i + time_window]:
                self.dic[self.cnt] = i
                self.cnt += 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        start = self.dic[index]
        end = start + self.time_window
        return self.eeg[start:end], self.eeg2[start:end], self.label[start]



class data_for_question_list():
    def __init__(self, file_path, file_list, feature, question, eval_video_ind, type='train'):
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for file in file_list:
            data_path = os.path.join(file_path, f'{str(file)}.mat')
            label_path = os.path.join(file_path, f'{str(file)}.xlsx')
            data = sio.loadmat(data_path)
            workbook = load_workbook(label_path)
            work_sheet = workbook.active
            third_row = []
            for cell in work_sheet[3]:
                third_row.append(cell.value)
            train_ind = []
            train_label = []
            eval_ind = []
            eval_label = []
            for i in range(12):
                if int(third_row[i * 6 + 1][-1]) != eval_video_ind:
                    train_ind.append(i + 1)
                    if int(third_row[i * 6 + 2 * question]) > 3:
                        train_label.append(0)
                    else:
                        train_label.append(1)
                else:
                    eval_ind.append(i + 1)
                    if int(third_row[i * 6 + 2 * question]) > 3:
                        eval_label.append(0)
                    else:
                        eval_label.append(1)       
            for i in range(len(train_ind)):
                eeg = data[f'video_{feature}_feature_{train_ind[i]}']
                X_train.append(eeg)
                for j in range(eeg.shape[0]):
                    y_train.append(train_label[i])
            for i in range(len(eval_ind)):
                eeg = data[f'video_{feature}_feature_{eval_ind[i]}']
                X_test.append(eeg)
                for j in range(eeg.shape[0]):
                    y_test.append(eval_label[i])
                    
        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        _, p, b = X_train.shape
        X_train = np.reshape(X_train, [X_train.shape[0], p * b])
        X_test = np.reshape(X_test, [X_test.shape[0], p * b])
        eeg_scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = eeg_scaler.transform(X_train)
        X_test = eeg_scaler.transform(X_test)

        if type == 'train':
            self.X = torch.from_numpy(X_train)
            self.y = torch.from_numpy(y_train)
            self.len = X_train.shape[0]
        else:
            self.X = torch.from_numpy(X_test)
            self.y = torch.from_numpy(y_test)
            self.len = X_test.shape[0]
    
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.X[item], self.y[item]

if __name__ == '__main__':
    file_list = [i for i in range(1, 9)]
    file_path = f'../data/new_data/mat_features_four_class/ai_high'
    eeg_test_dataset = data_for_question_list(file_path=file_path, file_list=file_list, feature='de',
                                          question=3, eval_video_ind=2, type='test')
    labels = [0, 0]
    for i in range(eeg_test_dataset.len):
        labels[int(eeg_test_dataset.y[i])] += 1
    print(labels)
        
        
