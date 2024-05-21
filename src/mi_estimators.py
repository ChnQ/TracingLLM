#!/usr/bin/env python
# Credit: Adapted and modifed from Greg Ver Steeg
# Or go to http://www.isi.edu/~gregv/npeet.html
import torch
import numpy as np


def distmat(X):
    """ distance matrix
    """
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def kernelmat(X, sigma, ktype='gaussian'):
    """ kernel matrix baker
    """
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))


    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(Kx, H)

    return Kxc


def hsic_normalized_cca(x, y, sigma=50., ktype='gaussian'):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma, ktype=ktype)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


def estimate_mi_hsic(x, y, t, ktype='gaussian'):
    estimate_IXT = hsic_normalized_cca(x, t, ktype=ktype)
    estimate_IYT = hsic_normalized_cca(y, t, ktype=ktype)
    return estimate_IXT, estimate_IYT