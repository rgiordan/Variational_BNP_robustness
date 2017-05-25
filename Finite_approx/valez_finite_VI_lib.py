"""
@author: Runjing Liu
"""

import autograd.numpy as np
import autograd.scipy as sp

import matplotlib.pyplot as plt
from copy import deepcopy
import math


# Data_shape: D,N,K

def phi_updates(nu, phi_mu, phi_var, X, sigmas, k):

    s_eps = sigmas['eps']
    s_A = sigmas['A']
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_mu)[1]


    phi_var[k] = (1/s_A + np.sum(nu[:, k]) / s_eps)**(-1)

    phi_summation = 0
    for n in range(N):
        phi_dum1 = X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n, k] * phi_mu[:, k]
        phi_summation += nu[n,k]*phi_dum1

    phi_mu[:,k] = (1 / s_eps) * phi_summation\
        * (1 / s_A + np.sum(nu[:, k]) / s_eps)**(-1)


def nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, n, k, digamma_tau):
    s_eps = sigmas['eps']
    s_A = sigmas['A']
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_mu)[1]

    nu_term1 = digamma_tau[k,0] - digamma_tau[k,1]

    nu_term2 = (1. / (2. * s_eps)) * (phi_var[k]*D + np.dot(phi_mu[:,k], phi_mu[:,k]))

    nu_term3 = (1./s_eps) * np.dot(phi_mu[:, k], X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n,k] * phi_mu[:, k])

    script_V = nu_term1 - nu_term2 + nu_term3

    nu[n,k] = 1./(1.+np.exp(-script_V))



def tau_updates(tau, nu, alpha):
    N = np.shape(nu)[0]
    K = np.shape(nu)[1]

    tau[:,0] = alpha/K + np.sum(nu,0)
    tau[:,1] = N  + 1 - np.sum(nu,0)


def cavi_updates(tau, nu, phi_mu, phi_var, X, alpha, sigmas):
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_mu)[1]

    assert np.shape(X)[0] == np.shape(nu)[0]
    assert np.shape(X)[1] == np.shape(phi_mu)[0]
    assert np.shape(nu)[1] == np.shape(phi_mu)[1]

    digamma_tau = sp.special.digamma(tau)

    for n in range(N):
        for k in range(K):
            nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, n, k, digamma_tau)

    for k in range(K):
        phi_updates(nu, phi_mu, phi_var, X, sigmas, k)

    tau_updates(tau, nu, alpha)


def compute_elbo(tau, nu, phi_mu, phi_var, X, sigmas, alpha):

    sigma_eps = sigmas['eps']
    sigma_A = sigmas['A']
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_mu)[1]

    digamma_tau = sp.special.digamma(tau)
    digamma_sum_tau = sp.special.digamma(tau[:,0] + tau[:,1])

    # bernoulli terms
    elbo_term1 = (alpha/K - 1) * np.sum( digamma_tau[:,0] - digamma_sum_tau[:] )

    elbo_term2 = np.sum(np.dot(nu, digamma_tau[:,0]) + np.dot(1-nu, digamma_tau[:,1])) \
                - N * np.sum(digamma_sum_tau)

    elbo_term3 = \
        -K*D/2.*np.log(2.*np.pi*sigma_A) - 1./(2.*sigma_A) *\
        (np.sum(phi_var)*D + np.trace(np.dot(phi_mu.T , phi_mu)))

    elbo_term4 = \
        -N * D/2. * np.log(2. * np.pi * sigma_eps)\
        -1/(2. * sigma_eps) * np.trace(np.dot(X.T, X)) \
        +1/(sigma_eps) * np.trace(np.dot(np.dot(nu, phi_mu.T), X.T))\
        -1/(2. * sigma_eps) * np.sum(np.dot(nu, D*phi_var + np.diag(np.dot(phi_mu.T, phi_mu)))) \
        -1/(2.*sigma_eps) * np.trace(np.dot(np.dot(nu, np.dot(phi_mu.T, phi_mu) - np.identity(K) \
                                                    * np.diag(np.dot(phi_mu.T, phi_mu))), nu.T))\

    # The log beta function is not in autograd's scipy.
    log_beta = sp.special.gammaln(tau[:,0]) + sp.special.gammaln(tau[:,1]) \
        - sp.special.gammaln(tau[:,0] + tau[:,1])

    elbo_term5 = np.sum(log_beta - \
        (tau[:,0] - 1) * digamma_tau[:,0] - \
        (tau[:,1] - 1) * digamma_tau[:,1] + \
        (tau[:,0] + tau[:,1] -2.) *  digamma_sum_tau[:])

    elbo_term6 = np.sum(1. / 2. * np.log((2. * np.pi * np.exp(1.) * \
        phi_var)**D))

    elbo_term7 = np.sum(np.sum( -np.log(nu ** nu) - np.log((1.-nu) ** (1.-nu)) ))

    elbo = elbo_term1 + elbo_term2 + elbo_term3 + elbo_term4 + elbo_term5 + elbo_term6 + elbo_term7

    return(elbo, elbo_term1, elbo_term2, elbo_term3, elbo_term4, elbo_term5, elbo_term6, elbo_term7)


def generate_data(Num_samples, D, K_inf, sigma_A, sigma_eps):
    Pi = np.ones(K_inf) * .8
    Z = np.zeros([Num_samples, K_inf])

    # Parameters to draw A from MVN
    mu = np.zeros(D)

    # Draw Z from truncated stick breaking process
    Z = np.random.binomial(1, Pi, [ Num_samples, K_inf ])

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


def generate_data(Num_samples, D, K_inf, sigma_A, sigma_eps):
    Pi = np.ones(K_inf) * .8
    Z = np.zeros([Num_samples, K_inf])

    # Parameters to draw A from MVN
    mu = np.zeros(D)

    # Draw Z from truncated stick breaking process
    Z = np.random.binomial(1, Pi, [ Num_samples, K_inf ])

    # Draw A from multivariate normal
    A = np.random.multivariate_normal(mu, sigma_A * np.identity(D), K_inf)

    # draw noise
    epsilon = np.random.multivariate_normal(
        np.zeros(D), sigma_eps*np.identity(D), Num_samples)

    # the observed data
    X = np.dot(Z,A) + epsilon

    return Pi, Z, mu, A, X
