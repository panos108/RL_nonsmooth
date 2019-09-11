import os
from os.path import join
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal
from Dynamic_system import specifications, DAE_system, integrator_model
from casadi import *
eps = np.finfo(np.float32).eps.item()

def select_action(control_mean, control_sigma):
    """
    Sample control actions from the distribution their distribution
    input: Mean, Variance
    Output: Controls, log_probability, entropy
    """
    s_cov = control_sigma.diag()**2


    dist = MultivariateNormal(control_mean, s_cov)
    control_choice = dist.sample()
    log_prob = dist.log_prob(control_choice)
    entropy = dist.entropy()
    return control_choice, log_prob, entropy




def run_episode(policy1, nk, dist, F, x0, u_min, u_max):
    """
    Single MC: Compute a single episode given a policy.
    input: Specification
    Output: Reward for each episode
    """
    #-----------Initialize-----------------------------------------------#
    container = [None for i in range(nk)]

    log_probs1 = container.copy()

    m1 = np.zeros([nk, 2])

    U = np.zeros([nk, 2])
    y = np.zeros([nk, 3])

    # define initial conditions
    t = 0.
    integrated_state = x0.copy()
    #---------------------------------------------------------------------#

    #------------------------ Perform MC for each time interval of (nk) in total
    for ind in range(nk):
        #--------------------- past inputs for neural network ------------#
        if ind>0:
            integrated_staten1 = integrated_staten.copy()#(integrated_state- np.array([9, 1000, 0.08]))/([9, 1000, 0.08])

        integrated_staten = (integrated_state- np.array([6, 400, 0.06]))/([6, 400, 0.06])
        if ind < 1:

            means  = [*np.zeros(2), *np.zeros(2), *integrated_staten]
        elif ind < 2:

            means  = [*m1[ind-1,:], *np.zeros(2), *integrated_staten1]
        else:

            means  = [*m1[ind-1,:], *m1[ind-2,:], *integrated_staten1]

        #--------------------------------------------------------------------#
        #-------------Compute next control action----------------------------#
        timed_state = Tensor((*integrated_staten, *means, ((240 - t)-120)/120))
        mean1, std1 = policy1(timed_state)
        action1, log_prob1, _ = select_action(mean1, std1)

        for i_u in range(len(u_min)):
            if action1[i_u] < u_min[i_u]:
                action1[i_u] = u_min[i_u]
            if action1[i_u] > u_max[i_u]:
                action1[i_u] = u_max[i_u]

        #---------------------------------------------------------------------#
        #------------------ Compute the next staes ---------------------------#

        xd = F(x0=vertcat(np.array(integrated_state)), p=vertcat(np.array(action1), dist))
        integrated_state = np.array(xd['xf'].T)[0]
        #---------------------------------------------------------------------#
        #---------------- storage data for next iter--------------------------#

        a_u = (u_max + u_min)/2
        b_u = (u_max - u_min)/2
        m1[ind, :] = (np.array(action1) - a_u)/b_u

        log_probs1[ind] = log_prob1

        U[ind, :]  = np.array(action1)
        y[ind, :] = integrated_state[:]



        t = t + 240./nk  # calculate next time
        #---------------------------------------------------------------------#
    #--------------Compute the reward-----------------------------------------#
    uu = U.reshape([2 * nk, 1])
    reward = integrated_state[2] \
            - (uu[2:]-uu[:-2]).T @ np.diagflat([3.125e-8, 3.125e-006]*(nk-1)) @ (uu[2:]-uu[:-2])
    #-------------------------------------------------------------------------#
    return reward.item(), log_probs1, U, y


def sample_episodes(policy1, optimizer1, sample_size, dun, F,
                    nk):

    """
    Perform all the MC's for one epoch.
    input: Policy, physical system, specifications
    output: expected reward, historical data
    """


    MC = sample_size
    rewards          =  [None for _ in range(sample_size)]
    summed_log_probs1 = [None for _ in range(sample_size)]

    log_prob_R1 = 0.0


    xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max,\
    states, algebraics, inputs, nd, na, nu, nmp, modparval = DAE_system()
    # ----------------------------------------
    for j in range(1):
        h_ys = np.zeros([sample_size, 13, 3])
        h_us = np.zeros([sample_size, 12, 2])

        for epi in range(sample_size):
            x0 = np.array([1., 150., 0.])
            dist = (np.random.multivariate_normal(np.zeros(dun.shape),
                                              np.diag(dun * 0.00))) # Parametric uncertainty (in this example there is 0
            x0 += np.random.multivariate_normal([0, 0, 0], np.diagflat([1e-3*np.array([1., 150.**2, 0.])])) # Random intial conditions
            #----------------------- Historical data ------------------------------------------------#
            h_ys[epi, 0, :] = x0
            reward, log_probs1, U, ys = run_episode(policy1, nk, dist, F, x0, u_min, u_max)
            h_ys[epi, 1:, :] = ys
            h_us[epi, :, :] = U
            rewards[epi] = reward

            summed_log_probs1[epi] = sum(log_probs1)
            #------------------------------------------------------------------------------------------#

    #------------- Compute expectation and std pf reward ----------------------------------------------#
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    #--------------------------------------------------------------------------------------------------#
    #------------------------------- Backward compute the baseline ------------------------------------#
    for epi in reversed(range(sample_size)):
        baselined_reward = (rewards[epi] - reward_mean) / (reward_std + eps)
        log_prob_R1 = log_prob_R1 - summed_log_probs1[epi] * baselined_reward

    mean_log_prob_R1 = log_prob_R1 / sample_size
    #---------------------------------------------------------------------------------------------------#
    return mean_log_prob_R1, reward_mean, reward_std, rewards, h_us, h_ys


def training(policy1, optimizer1, epochs, epoch_episodes, dun, F,
             nk, path1):
    """
    Run the full training with MCs and update policy
    input: initial policy, physcical system and its specifications
    output: Historical data, trained policy 
    """""
    #-------------------Initialize matrices----------------------------#
    his_rewards = np.zeros([epochs, epoch_episodes])


    his_u = np.zeros([epochs, epoch_episodes, nk, 2])
    his_y = np.zeros([epochs, epoch_episodes, nk+1, 3])

    # prepare directories for results
    os.makedirs('figures', exist_ok=True)
    os.makedirs('serializations', exist_ok=True)

    rewards_record = []
    rewards_std_record = []
    #----------------------------------------------------------------#
    print(f"Training for {epochs} iterations of {epoch_episodes} sampled episodes each!")
    for epoch in range(epochs):

        # get the log probability of the policy and the expected reward after all the episodes
        mean_log_prob1, reward_mean, reward_std, rewards, uu, ys = sample_episodes(
            policy1, optimizer1, epoch_episodes, dun, F,
        nk)
        #------------------- Optimize Policy------------------------#
        optimizer1.zero_grad()
        mean_log_prob1.backward()
        optimizer1.step()
        #-----------------------------------------------------------#
        #-------------- Sve data for each epoch-epiode-------------#
        his_rewards[epoch, :] = rewards
        his_u[epoch, :, :, :] = uu
        his_y[epoch, :, :, :] = ys


        rewards_record.append(reward_mean)
        rewards_std_record.append(reward_std)
        #-----------------------------------------------------------#
        print('epoch:', epoch)
        print(f'mean reward: {reward_mean:.3} +- {reward_std:.2}')

        torch.save(policy1, (path1+'/policyplant11'+str(epoch)+str(int(1000*reward_mean))+'.pt')) #SAVE POLICY FOR EACH EPOCH
    # --------------------- Save all Data----------------------#
    import pickle
    pickle.dump(his_rewards, open(path1+'/rew.p', 'wb'))

    pickle.dump(his_u, open(path1+'/u.p', 'wb'))
    pickle.dump(his_y, open(path1+'/y.p', 'wb'))
    # ---------------------------------------------------------#
    return rewards_record, rewards_std_record
