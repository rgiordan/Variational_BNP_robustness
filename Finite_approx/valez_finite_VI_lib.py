# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:10:41 2017

@author: Haiying Liang
"""

# import numpy as np
# import scipy as sp
import autograd.numpy as np
import autograd.scipy as sp
#from sp import special
from autograd.scipy import special
import matplotlib.pyplot as plt
from copy import deepcopy
#import math

# from beta_process_vb_lib import *

# Data_shape: D,N,K

def phi_updates(nu, phi_mu, phi_var, X, sigmas, Data_shape):

    s_eps = sigmas['eps']
    s_A = sigmas['A']
    D = Data_shape['D']
    N = Data_shape['N']
    K = Data_shape['K']

    for k in range(K):
        phi_var[k] = (1/s_A + np.sum(nu[:, k]) / s_eps)**(-1)

        phi_summation = 0
        for n in range(N):
            phi_dum1 = X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n, k] * phi_mu[:, k]
            phi_summation += nu[n,k]*phi_dum1

        #    dum1 = 0
        #    for l in range(K):
        #        if (l != k):
        #            dum1 += nu[n,l] * phi_mu[:,l]

        #    phi_summation += nu[n,k] * (X[n,:] - dum1)

        phi_mu[:,k] = (1 / s_eps) * phi_summation\
                 * (1 / s_A + np.sum(nu[:, k]) / s_eps)**(-1)


def get_nu(tau, phi_mu, phi_var, X, sigmas, Data_shape):
    s_eps = sigmas['eps']
    K = Data_shape['K']
    N = Data_shape['N']
    D = Data_shape['D']

    nu_term1 = [[ sp.special.digamma(tau[k, 0]) - sp.special.digamma(tau[k, 1])
                  for n in range(N) ] for k in range(K) ]
    nu_term2 = [[ (1. / (2. * s_eps)) * (phi_var[k] * D + np.dot(phi_mu[:, k], phi_mu[:, k]))
                  for n in range(N) ] for k in range(K) ]
    nu_term1 = [[ (1./s_eps) * np.dot(phi_mu[:, k], X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n,k] * phi_mu[:, k])
                  for n in range(N) ] for k in range(K) ]

    return 1. / (1. + np.exp(-(nu_term1 - nu_term2 + nu_term3)))


def nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape):

    s_eps = sigmas['eps']
    K = Data_shape['K']
    N = Data_shape['N']
    D = Data_shape['D']

    for n in range(N):
        for k in range(K):
            nu_term1 = sp.special.digamma(tau[k, 0]) - sp.special.digamma(tau[k, 1])
            nu_term2 = (1. / (2. * s_eps)) * (phi_var[k] * D + np.dot(phi_mu[:, k], phi_mu[:, k]))
            nu_term3 = (1./s_eps) * np.dot(phi_mu[:, k], X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n,k] * phi_mu[:, k])

            script_V = nu_term1 - nu_term2 + nu_term3

            nu[n,k] = 1. / (1. + np.exp(-script_V))



def tau_updates(tau, nu, alpha, Data_shape):
    K = Data_shape['K']
    N = Data_shape['N']

    tau[:,0] = alpha/K + np.sum(nu,0)
    tau[:,1] = N  + 1 - np.sum(nu,0)


def Elbo(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, alpha):

    sigma_eps = sigmas['eps']
    sigma_A = sigmas['A']
    D = Data_shape['D']
    N = Data_shape['N']
    K = Data_shape['K']

    # bernoulli terms
    elbo_term1 = np.sum( np.log(alpha/K) + (alpha/K - 1)*(sp.special.digamma(tau[k,0]) \
                  - sp.special.digamma(tau[k,0] + tau[k, 1])) for k in range(K))

    elbo_term2 = 0
    for k in range(K):
        for n in range(N):
            elbo_term2 += nu[n,k] * sp.special.digamma(tau[k,0]) + (1-nu[n,k])*\
                    sp.special.digamma(tau[k,1]) \
                    - sp.special.digamma(tau[k,0]+tau[k,1])

    elbo_term3 = np.sum(-D/2*np.log(2*np.pi*sigma_A) - 1/(2*sigma_A) *\
        (phi_var[k]*D + \
        np.dot(phi_mu[:, k] , phi_mu[:, k])) for k in range(K) )

    elbo_term4 = 0
    for n in range(N):
        summ1 = np.sum(nu[n,k] * np.dot(phi_mu[:,k], X[n,:]) for k in range(K))
        summ2 = np.sum(
            np.sum(
                nu[n,k1] * nu[n,k2] * np.dot(phi_mu[:,k1], phi_mu[:,k2]) \
                for k1 in range(k2)) for k2 in range(K))
        summ3 = np.sum(nu[n,k] * (D*phi_var[k] + \
            np.dot(phi_mu[:,k], phi_mu[:,k])) for k in range(K))

        elbo_term4 += - D / 2 * np.log(2 * np.pi * sigma_eps) - \
            1 / (2 * sigma_eps) * (
                np.dot(X[n,:], X[n,:]) - 2 * summ1 + 2 * summ2 + summ3)

    elbo_term5 = \
        np.sum(sp.special.gammaln(tau[:,0]) + sp.special.gammaln(tau[:,1]) - \
               sp.special.gammaln(tau[:,0] + tau[:,1]) - \
        (tau[:,0] - 1) * sp.special.digamma(tau[:,0]) - \
        (tau[:,1] - 1) * sp.special.digamma(tau[:,1]) + \
        (tau[:,0] + tau[:,1] -2) *  sp.special.digamma(tau[:,0] + tau[:,1]))

    elbo_term6 = np.sum(1 / 2 * np.log((2 * np.pi * np.exp(1)) ** D * \
        phi_var[k]**D) for k in range(K))

    elbo_term7 = np.sum(np.sum( -np.log(nu ** nu) - np.log((1-nu) ** (1-nu)) ))

    elbo = elbo_term1 + elbo_term2 + elbo_term3 + elbo_term4 + elbo_term5 + elbo_term6 + elbo_term7

    return(elbo, elbo_term1, elbo_term2, elbo_term3, elbo_term4, elbo_term5, elbo_term6, elbo_term7)


def generate_data(Num_samples, D, K_inf, sigma_A, sigma_eps):
    Pi = np.ones(K_inf) * .8
    Z = np.zeros([Num_samples, K_inf])

    # Parameters to draw A from MVN
    mu = np.zeros(D)
    # Draw Z from truncated stick breaking process
    for k in range(K_inf):
        for n in range(Num_samples):
            Z[n,k] = np.random.binomial(1, Pi[k])

    # Draw A from multivariate normal
    A = np.random.multivariate_normal(mu, sigma_A * np.identity(D), K_inf)

    # draw noise
    epsilon = np.random.multivariate_normal(
        np.zeros(D), sigma_eps*np.identity(D), Num_samples)

    # the observed data
    X = np.dot(Z,A) + epsilon

    return Pi, Z, mu, A, X


def initialize_parameters(Num_samples, D, K_approx):
    tau = np.random.uniform(0, 1, [K_approx, 2]) # tau1, tau2 -- beta parameters for v
    nu =  np.random.uniform(0, 1, [Num_samples, K_approx]) # Bernoulli parameter for z_nk

    phi_mu = np.random.normal(0, 1, [D, K_approx]) # kth mean (D dim vector) in kth column
    phi_var = np.ones(K_approx)

    return tau, nu, phi_mu, phi_var
