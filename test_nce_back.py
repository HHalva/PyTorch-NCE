from __future__ import print_function

import torch as th
from torch import nn
import torchvision
import torch.nn.functional as F
import pdb




#test NCE implementation
def nce_test():
    """
    Test implementation of NCE for Gaussian
    """
    #specify data size
    data_dim = 5
    Td = 1000
    noise_ratio = 50
    Tn = Td*noise_ratio
    Td_batch = 200
    Tn_batch = Td_batch*noise_ratio

    #create Pd and create artificial data
    cov_base = th.rand((data_dim, data_dim))
    cov_mat = (cov_base+cov_base.t())/2  + 5*th.eye(data_dim)
    p_data = th.distributions.multivariate_normal.MultivariateNormal(
        th.zeros(data_dim), cov_mat)
    data_labels = th.ones(Td)
    data_sample = th.utils.data.TensorDataset(p_data.sample((Td,)),
                                              data_labels)
    data_loader = th.utils.data.DataLoader(data_sample,
                                           batch_size=Td_batch,
                                           shuffle=True)

    #specify noise parameters for later
    noise_cov_mat = th.eye(data_dim)

    #set up 'model' to be estimated
    model_log_P = LogNormGauss(5)

    #set up optimization parameters
    start_epoch = 0
    end_epoch = 1000
    start_lr = 0.01
    decay_freq = 50
    decay_gamma = 0.1
    optimizer = th.optim.Adam(model_log_P.parameters(), lr=start_lr)
    lr_sched = th.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=decay_freq,
                                            gamma=decay_gamma)

    #train
    for epoch in range(start_epoch, end_epoch):
        lr_sched.step()
        model_log_P.train()
        for i, (data_batch, data_labels) in enumerate(data_loader):
            #sample noise data for current input batch
            noise_distr = th.distributions.multivariate_normal.MultivariateNormal(
                th.zeros(data_dim), noise_cov_mat)
            noise_batch = noise_distr.sample((Tn_batch,))
            noise_labels = th.zeros(Tn_batch)
            #combine data and noise samples
            joint_batch = th.cat((data_batch, noise_batch), 0)
            joint_labels = th.cat((data_labels, noise_labels), 0)

            #forward pass             
            log_P_data = model_log_P(joint_batch)
            log_P_noise = noise_distr.log_prob(joint_batch)
            log_P_diff = log_P_noise-log_P_data
            loss = NCE_loss(log_P_diff, joint_labels, Td_batch, noise_ratio)
            print(loss)
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pdb.set_trace()


def NCE_loss(log_diff, labels, Td, noise_ratio):
    """
    Negative of the objective function for NCE training with a custom, stable,
    implementation of the custom sigmoids with log difference input
    """
    u = log_diff
    y = labels
    v = th.tensor(noise_ratio, dtype=th.float)
    u_switch = th.clamp(u, min=0)
    v_switch = th.clamp(-u/th.abs(u), min=0)
    loss = u_switch - y*u + (y-1)*th.log(v) + th.log(
        v**v_switch + th.exp(-th.abs(u))*v**(1-v_switch))

    return th.sum(loss)/Td


class LogNormGauss(nn.Module):
    """
    Log of Un-normalized Gaussian calculated simultaneously for a batch
    of multi-dimensional observations
    """
    def __init__(self, in_dim):
        super().__init__()
        self.W = th.Tensor(in_dim, in_dim)
        nn.init.normal_(self.W)
        self.L = th.tril(self.W)+th.tril(th.eye(5, 5))
        self.L = nn.Parameter(self.L)
        self.c = nn.Parameter(th.Tensor([0.01]))

    def forward(self, X):
        #Note: since we have several multi-di obs we only want the diagonal

        return th.diag(-0.5 * th.matmul(X, th.matmul(self.W, X.t())) + self.c)


#RUN
if __name__ == '__main__':

    nce_test()



