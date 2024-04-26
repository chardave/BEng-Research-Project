import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

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
            #print(q)

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
    
    # normalise inputs?

class NeuralNetwork(nn.Module):

    def __init__(self, ninput):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(ninput, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 7,bias = False)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
      

def main() :

    start = time.time()


    dataset = FeatureDataset()

    # split the data set into training and testing data
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% of the data for training
    test_size = dataset_size - train_size  # Remaining data for testing

    training_data, test_data = random_split(dataset, [train_size, test_size])

    # Create data loaders
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size = batch_size)

    for Q, Tau in train_dataloader:
        print(f"Shape of X [N, C]: {Q.shape}")
        print(f"Shape of y: {Tau.shape} {Tau.dtype}")
        print(Q)
        print(Tau)
        break

    # create neural network function
    model = NeuralNetwork(Q.shape[1])
    #print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

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
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(Q)
                total_loss += loss
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            # plot the prediction agaist tau

        model.eval()

        train_loss = 0.0
        for i in range(train_size):
            q, tau = training_data[i]
            p = model(q)
            train_loss += loss_fn(p, tau).item()

        train_loss /= float(train_size) # normalizes the MSE wrt to the number of samples

        test_loss = 0.0
        for i in range(test_size):
            q, tau = test_data[i]
            p = model(q)
            test_loss += loss_fn(p, tau).item()

        test_loss /= float(test_size) # normalizes the MSE wrt to the number of samples

        # fig, ax = plt.subplots(7,1, sharex=True, tight_layout=True)
        # for i in range(7):
        #     ax[i].plot(df['t'],p[:,i], ':', label=f'tau_estimated_{i}')
        #     df.plot(x='t', y=f'tau_{i}',ax=ax[i])
        #     plt.show()


        return train_loss, test_loss

    epochs = 250

    train_losses = []
    test_losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, test_loss = train(train_dataloader, model, loss_fn, optimizer)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    print(f"Final Train Loss: {train_loss:>7f}")
    print(f"Final Test Loss: {test_loss:>7f}")

    print("Done!")

    model.eval()

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(range(epochs), train_losses, label='train')
    ax.plot(range(epochs), test_losses, label='test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSELoss')
    ax.legend()
    ax.grid()
    ax.set_title('Neural Network ')
    fig.savefig("lossepoch_q_spt.png")

    train_loss = 0.0
    test_loss = 0.0
    for i in range(train_size):
        q, tau = training_data[i]
        p = model(q)
        train_loss += loss_fn(p, tau).item()

    train_loss /= float(train_size) # normalizes the MSE wrt to the number of samples

    for i in range(test_size):
        q, tau = test_data[i]
        p = model(q)
        test_loss += loss_fn(p, tau).item()

    test_loss /= float(test_size) # normalizes the MSE wrt to the number of samples

    # plot prediction against tau
    dt = 0.01  # 100Hz
    df = pd.read_csv('init_data_spt.csv')
    df['t'] = np.linspace(0, dt*df.shape[0], df.shape[0])
    preds = []
    
    evalloss = nn.L1Loss()
    final_loss = 0.0
    
    for t in range(df.shape[0]):
        qcols = [f'q_{i}' for i in range(7)]
        taucols = [f'tau_{i}' for i in range(7)]
        tau = torch.tensor(df[taucols].iloc[t], dtype = torch.float32)
        q = torch.tensor(df[qcols].iloc[t], dtype = torch.float32)
        p = model(q)
        final_loss += evalloss(p, tau).item()
        preds.append(p.detach().cpu().numpy().flatten().tolist())
    
    final_loss /= float(df.shape[0])

    print(f"Final Evaluation Loss: {final_loss:>7f}")
    
    preds = np.array(preds)

    df['t'] = np.linspace(0, dt*df.shape[0], df.shape[0])

    fig, ax = plt.subplots(7, 1, sharex=True, tight_layout=True)

    plt.rcParams['font.size']='16'


    for i in range(7):
        ax[i].plot(df['t'], preds[:, i], ':', label=f'tau_estimated_{i}')
        df.plot(x= 't', y=f'tau_{i}', ax=ax[i])
        ax[i].legend().remove()
        ax[i].set_ylabel(f'joint {i}', fontsize=12)
        for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            label.set_fontsize(14)
    
    fig.suptitle('Comparing Prediction Torques and True Torques')
    
    end = time.time()
    total_time = end-start
    print(f"Time taken to run: {total_time}" )

    fig.savefig("model_q_spt.png")
    plt.show()

    # Save NN
    torch.save(model, 'model_q_spt.nn')

    





    return 0

if __name__ == "__main__":
    main()
