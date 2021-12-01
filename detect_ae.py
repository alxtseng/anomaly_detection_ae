import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models as models
import os
import argparse


class ACCDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.X[idx]


class AutoEncoder(nn.Module):
    def __init__(self, n_feature, gpu):
        super(AutoEncoder, self).__init__()
        self.gpu = gpu
        self.n_feature = n_feature
        self.n_hidden1 = 100
        self.n_latent = 10
        self.n_hidden2 = self.n_hidden1
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feature, self.n_hidden1),
            nn.ReLU(),
            nn.Linear(self.n_hidden1, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden2),
            nn.ReLU(),
            nn.Linear(self.n_hidden2, self.n_feature)
        )
        if self.gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x_e = self.encoder(x)
        x_d = self.decoder(x_e)
        return x_d

    def loss(self, pred_x, true_x):
        if self.gpu:
            true_x = true_x.cuda()
        return torch.norm(pred_x - true_x.float(), p=2, dim=1)


C = 50  # Number cars
F = 2  # Number of Features
T = 2000  # Number Timesteps
# data_paths = {'train': '', 'test_m': '', 'test_b': ''}
data_path = './data'
path_heads = {'train': 'train_benign', 'test_m': 'test_mal', 'test_b': 'test_benign'}


def load_data(data_path, path_head):
    xs = []
    files = os.listdir("%s/%s" %(data_path, path_head))
    for file in files:
        if file[-4:] != ".csv":
            continue
        # print('loading file: ', file)
        file_path = "%s/%s/" %(data_path, path_head) + file
        x = np.loadtxt(file_path)
        xs.append(x)
    N_ = len(xs)
    _X = np.zeros((N_, T * C * F))
    for n in range(N_):
        X = xs[n]
        _X[n, :] = X.flatten()
    return _X  # shape (N, T, (C*F))


def train_epoch(model, optimizer, dataloader):
    model.train()

    cum_loss = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        #         if (B==1):
        #             continue
        pred = model(X.float())
        loss = torch.sum(model.loss(pred, y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
        # pred_c = pred.max(1)[1].cpu()
        tot_num = tot_num + B

    return cum_loss / tot_num


def eval_data(model, dataloader):
    model.eval()
    tot = 0
    ls = []
    for X, y in dataloader:
        pred = model(X.float())
        loss = model.loss(pred, y)
        ls += list(loss.detach().numpy())
        #         exs += np.sum(loss.detach().numpy() > threshold)
        tot += X.size()[0]
    #     print("# of samples exceeding reconstruction loss threshold: %d, # of total: %d" %(exs, tot))
    return ls, tot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int)
    args = parser.parse_args()

    # train_X = load_Xs(data_path=data_path)
    train_X = load_data(data_path=data_path, path_head=path_heads['train'])
    eval_X_b = load_data(data_path=data_path, path_head=path_heads['test_b'])
    eval_X_m = load_data(data_path=data_path, path_head=path_heads['test_m'])

    print(train_X.shape)
    print(eval_X_b.shape)
    print(eval_X_m.shape)
    n_feature = train_X.shape[1]

    trainset = ACCDataset(train_X)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)
    evalset_b = ACCDataset(eval_X_b)
    evalloader_b = torch.utils.data.DataLoader(evalset_b, batch_size=10, shuffle=False)
    evalset_m = ACCDataset(eval_X_m)
    evalloader_m = torch.utils.data.DataLoader(evalset_m, batch_size=10, shuffle=False)

    ae_model = AutoEncoder(n_feature=n_feature, gpu=0)
    print('Model Initialized.')
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3, weight_decay=1e-5)

    for e in range(args.n_epoch):
        l = train_epoch(model=ae_model, optimizer=optimizer, dataloader=trainloader)
        if e % 10 == 0:
            print("Epoch %d" % (e), "loss %f" % (l))
            # calc reconstruction loss
            # threshold = l
            # print("Train")
            train_ls, train_tot = eval_data(ae_model, dataloader=trainloader)
            sorted_train_ls = sorted(train_ls)
            # print("Eval benign")
            eval_ls_b, eval_tot_b = eval_data(ae_model, dataloader=evalloader_b)
            # print("Eval mal")
            eval_ls_m, eval_tot_m = eval_data(ae_model, dataloader=evalloader_m)

            # AUC score
            from sklearn import metrics
            y = np.array([0 for _ in range(eval_tot_b)]+[1 for _ in range(eval_tot_m)])
            pred = np.array(eval_ls_b+eval_ls_m)
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            print("The auc score is", auc)




