import torch.nn.functional as F
from copy import deepcopy
import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from tqdm import tqdm

class MLPTrainer():
    def __init__(self, mlp, optimizer, scheduler, device, print_freq=100):
        self.mlp = mlp
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.print_freq = print_freq
        self.device = device

        self.steps = 0
        self.epoch_loss_history = []
        self.max_accuracy = 0
        self.max_model = None
        self.is_save_model = False

    def train(self, data_loader, test_loader, epochs):
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.
            self.mlp.train()
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()
                eeg, y = data
                eeg = eeg.float().to(self.device)
                y = y.float().to(self.device)
                pred = self.mlp(eeg=eeg)

                loss = self._loss(pred, y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

            with torch.no_grad():
                self.mlp.eval()
                predicted = []
                labels = []
                for eeg, y in test_loader:
                    eeg = eeg.float().to(self.device)
                    y = y.float()
                    pred = self.mlp(eeg=eeg)
                    pred = torch.argmax(pred, dim=1)
                    pred = pred.cpu().numpy()
                    predicted.append(pred)
                    labels.append(y)
                accuracy = balanced_accuracy_score(np.concatenate(labels), np.concatenate(predicted))
                if accuracy > self.max_accuracy:
                    self.max_accuracy = accuracy
                    self.max_model = deepcopy(self.mlp.state_dict())

    def _loss(self, p_y_pred, y_target):
        return F.cross_entropy(p_y_pred, y_target.long())