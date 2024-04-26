import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

# class to extract the data from the csv file, outputing a tensor for the positions and torques
class FeatureDataset(Dataset) :

    def __init__(self) :

        dt = 0.01  # 100Hz
        df = pd.read_csv('init_data_spt.csv')
        df['t'] = np.linspace(0, dt*df.shape[0], df.shape[0])

        for i in range(7):
            q = df[f'q_{i}']
            t = df['t']
            dq = np.gradient(q, t)
            df[f'dq_{i}'] = dq
            df[f'ddq_{i}'] = np.gradient(dq, t)

        self.df = df
        self.dt = dt

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        qcols = [f'q_{i}' for i in range(7)]
        q = torch.tensor(self.df[qcols].iloc[idx], dtype = torch.float32)

        taucols = [f'tau_{i}' for i in range(7)]
        tau = torch.tensor(self.df[taucols].iloc[idx], dtype = torch.float32)
        return q, tau
    
class LSTMNetwork(nn.Module):
    def __init__(self, ninput):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = ninput,
            hidden_size = 512,
            num_layers = 1,
            batch_first = True
        )
        self.linear = nn.Linear(512,7)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def main():

    start = time.time()

    dataset = FeatureDataset()

    # split the data set into training and testing data
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% of the data for training
    test_size = dataset_size - train_size  # Remaining data for testing
    
    training_data, test_data = random_split(dataset, [train_size, test_size])

    # Create data loaders
    batch_size = 64
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size = batch_size)

    for Q, Tau in train_dataloader:
        print(f"Shape of X [N, C]: {Q.shape}")
        print(f"Shape of y: {Tau.shape} {Tau.dtype}")
        break

    # create lstm model
    model = LSTMNetwork(Q.shape[1])

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # function to train and test the data that will be done at each epoch
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        total_loss = 0
        for batch, (Q, Tau) in enumerate(dataloader):

            # Compute prediction error
            pred = model(Q)
            loss = loss_fn(pred, Tau)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(Q)
                total_loss += loss
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            # plot the prediction agaist tau

        model.eval()

        train_loss = 0.0
        q_train = []
        tau_train = []
        for i in range(train_size):
            q, tau = training_data[i]
            q_train.append(q)
            tau_train.append(tau)

        q_train = torch.stack(q_train)
        tau_train = torch.stack(tau_train)
        #print(q_train)
        p_train = model(q_train)
        train_loss = loss_fn(p_train, tau_train)
        train_loss = train_loss.detach().numpy()

        test_loss = 0.0
        q_test = []
        tau_test = []
        for i in range(test_size):
            q, tau = test_data[i]
            q_test.append(q)
            tau_test.append(tau)

        q_test = torch.stack(q_test)
        tau_test = torch.stack(tau_test)
        #print(q_test)
        p_test = model(q_test)
        test_loss = loss_fn(p_test, tau_test)
        test_loss = test_loss.detach().numpy()

        return train_loss, test_loss

    epochs = 500
    
    train_losses = []
    test_losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, test_loss = train(train_dataloader, model, loss_fn, optimizer)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    print("Done!")
    print(f"Final Train Loss: {train_loss:>7f}")
    print(f"Final Test Loss: {test_loss:>7f}")

    model.eval()

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(range(epochs), train_losses, label='train')
    ax.plot(range(epochs), test_losses, label='test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSELoss')
    ax.legend()
    ax.grid()
    ax.set_title('LSTM')
    fig.savefig("LSTMlossepoch_q.png")

    # plot prediction against tau
    dt = 0.01  # 100Hz
    df = pd.read_csv('init_data_spt.csv')
    df['t'] = np.linspace(0, dt*df.shape[0], df.shape[0])
    preds = []
    qs = []
    taus = []

    evalloss = nn.L1Loss()
    final_loss = 0.0

    for t in range(df.shape[0]):
        qcols = [f'q_{i}' for i in range(7)]
        taucols = [f'tau_{i}' for i in range(7)]
        tau = torch.tensor(df[taucols].iloc[t], dtype = torch.float32)
        q = torch.tensor(df[qcols].iloc[t], dtype = torch.float32)
        qs.append(q)
        taus.append(tau)

    qs = torch.stack(qs)
    taus = torch.stack(taus)
    preds = model(qs)
    final_loss = evalloss(preds, taus)
    final_loss = final_loss.detach().numpy()

    print(f"Final Evaluation Loss: {final_loss:>7f}")

    # print(len(preds))

    df['t'] = np.linspace(0, dt*df.shape[0], df.shape[0])

    fig, ax = plt.subplots(7, 1, sharex=True, tight_layout=True)

    plt.rcParams['font.size']='16'

    for i in range(7):
        ax[i].plot(df['t'], preds[:, i].detach().numpy(), ':', label=f'tau_estimated_{i}')
        df.plot(x= 't', y=f'tau_{i}', ax=ax[i])
        ax[i].legend().remove()
        ax[i].set_ylabel(f'joint {i}', fontsize=12)
        for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            label.set_fontsize(14)

    fig.suptitle('Comparing Prediction Torques and True Torques')

    end = time.time()
    total_time = end-start
    print(f"Time taken to run: {total_time}" )

    fig.savefig("LSTMmodel_q.png")
    plt.show()

    # Save NN
    torch.save(model, 'LSTMmodel_q.nn')

    return 0

if __name__ == "__main__":
    main()

    

