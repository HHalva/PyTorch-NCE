from __future__ import print_function

import torch as th
from torch import nn
import torchvision
import torch.nn.functional as F
from sklearn.datasets import make_spd_matrix

from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import pdb
import numpy as np
from unnorm_mv_gauss import UnnormMVGaussian


#test NCE implementation
def nce_test():
    """
    Test implementation of NCE for Gaussian
    """
    #specify data size
    data_dim = 5
    Td = 100000
    noise_ratio = 50
    Tn = Td*noise_ratio
    Td_batch = 1000
    Tn_batch = Td_batch*noise_ratio

    #create Pd and create artificial data
    cov_base = th.tensor(make_spd_matrix(data_dim), dtype=th.float)
    tril_mat = th.tril(cov_base)
    cov_mat = th.matmul(tril_mat, tril_mat.t())
    true_c = -0.5*th.log(th.abs(
        th.det(cov_mat)))-(data_dim/2)*th.log(2*th.tensor(np.pi))
    p_data = MVN(th.zeros(data_dim), scale_tril=tril_mat)
    data_labels = th.ones(Td)
    data_sample = th.utils.data.TensorDataset(p_data.sample((Td,)),
                                              data_labels)
    data_loader = th.utils.data.DataLoader(data_sample,
                                           batch_size=Td_batch,
                                           shuffle=True)

    #specify noise parameters for later use
    noise_cov_mat = th.eye(data_dim)

    #set up the model to be estimated
    cov_model = th.tensor(make_spd_matrix(data_dim), dtype=th.float)
    tril_mat_model = th.tril(cov_model)
    model = UnnormMVGaussian(th.zeros(data_dim), scale_tril=tril_mat_model)
    model.scale_tril.requires_grad = True 
    model.normalizing_constant.requires_grad = True

    #set up optimization parameters
    start_epoch = 0
    end_epoch = 1000
    start_lr = 0.001
    momentum = 0.9
    decay_epochs = [50, 100, 250, 500, 750]
    decay_gamma = 0.1
    optimizer = th.optim.Adam([model.scale_tril, model.normalizing_constant],
                              lr=start_lr)
    lr_sched = th.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=decay_epochs,
                                                 gamma=decay_gamma)

    #train
    for epoch in range(start_epoch, end_epoch):
        lr_sched.step()
        print(epoch)
        for i, (data_batch, data_labels) in enumerate(data_loader):
            #sample noise data for current input batch
            noise_distr = MVN(th.zeros(data_dim), noise_cov_mat)
            noise_batch = noise_distr.sample((Tn_batch,))
            noise_labels = th.zeros(Tn_batch)
            #combine data and noise samples
            joint_batch = th.cat((data_batch, noise_batch), 0)
            joint_labels = th.cat((data_labels, noise_labels), 0)

            #forward pass             
            log_P_model = model.log_prob(joint_batch)
            log_P_noise = noise_distr.log_prob(joint_batch)
            log_P_diff = log_P_model - log_P_noise + 1e-20
            loss = NCE_loss(log_P_diff, joint_labels, Td_batch, noise_ratio)
            print(loss.item(), true_c.item(), model.normalizing_constant.item())
            print(F.mse_loss(model.scale_tril, p_data.scale_tril))

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_cov_mat = th.chain_matmul(model.scale_tril.detach(),
                                            model.scale_tril.detach().t())

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



#RUN
if __name__ == '__main__':

    nce_test()



