import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import gpytorch as gp
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

# class forming the gaussian process model
class MultitaskGPModel(gp.models.ExactGP):
    def __init__(self, x_input, y_input, likelihood):
        super(MultitaskGPModel,self).__init__(x_input,y_input,likelihood)
        self.mean_module = gp.means.MultitaskMean(gp.means.ConstantMean(), num_tasks=7)
        self.covar_module = gp.kernels.MultitaskKernel(gp.kernels.RBFKernel(), num_tasks=7, rank = 1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
def main():
    start = time.time()

    dataset = FeatureDataset()

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% of the data for training
    test_size = dataset_size - train_size  # Remaining data for testing

    training_data, test_data = random_split(dataset, [train_size, test_size])

    # Create data loaders
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size = batch_size)

    for Q, Tau in train_dataloader:
        print(f"Shape of Q [N, C]: {Q.shape}")
        print(f"Shape of TAU: {Tau.shape} {Tau.dtype}")
        break

    likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks = 7)
    model = MultitaskGPModel(Q, Tau, likelihood)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = gp.mlls.ExactMarginalLogLikelihood(likelihood,model)

    def train(dataloader,model, loss_fn, optimizer):
        
        model.train()
        likelihood.train()
        
        optimizer.zero_grad()
        pred = model(Q)
        loss = -loss_fn(pred, Tau)
        loss.backward()
        optimizer.step()

        model.eval()
        likelihood.eval()
        

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
        
    epochs = 200
    train_losses = []
    test_losses = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, test_loss = train(train_dataloader, model, loss_fn, optimizer)
        train_losses.append(-train_loss)
        test_losses.append(-test_loss)

    print('Done!')

    print(f"Final Train Loss: {-train_loss:>7f}")
    print(f"Final Test Loss: {-test_loss:>7f}")

    model.eval()

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(range(epochs), train_losses, label='train')
    ax.plot(range(epochs), test_losses, label='test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSELoss')
    ax.legend()
    ax.grid()
    fig.savefig("gplossepoch_q.png")



    # plot prediction against tau
    dt = 0.01  # 100Hz
    df = pd.read_csv('init_data_spt.csv')
    df['t'] = np.linspace(0, dt*df.shape[0], df.shape[0])

    qs = []
    taus = []
    

    for t in range(df.shape[0]):
        qcols = [f'q_{i}' for i in range(7)]
        taucols = [f'tau_{i}' for i in range(7)]
        tau = torch.tensor(df[taucols].iloc[t], dtype = torch.float32)
        q = torch.tensor(df[qcols].iloc[t], dtype = torch.float32)
        qs.append(q)
        taus.append(tau)

    qs = torch.stack(qs)
    taus = torch.stack(taus)
    preds = likelihood(model(qs))
    # preds.detach().cpu().numpy().flatten().tolist()

    final_loss = loss_fn(preds, taus)
    #final_loss = test_loss.detach().numpy() 

    print(f"Final Evaluation Loss: {final_loss:>7f}") 



    #print(preds)

    mean = preds.mean

    #lower, upper = preds.confidence_region()
    
    f, ax = plt.subplots(7, 1, sharex=True, tight_layout=True)

    plt.rcParams['font.size']='16'

    for i in range(7):
        ax[i].plot(df['t'],mean[:,i].detach().numpy(),':', label = f'tau_estimated_{i}')
        df.plot(x='t',y=f'tau_{i}', ax=ax[i])
        ax[i].legend().remove()
        ax[i].set_ylabel(f'joint {i}', fontsize=12)
        for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            label.set_fontsize(14)
        #ax[i].fill_between(df['t'],lower[:,i].detach().numpy(),upper[:,i].detach().numpy(), alpha=0.5)
    fig.suptitle('Comparing Prediction Torques and True Torques')
    

    end = time.time()
    total_time = end-start
    print(f"Time taken to run: {total_time}" )

    f.savefig("gpmodel_q.png")
    plt.show()

    # save GP Model
    torch.save(model, 'GPmodel_q.nn') 


    return 0

if __name__ == "__main__":
    main()

    
