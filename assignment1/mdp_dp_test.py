#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mdp_dp import *
import gym
import sys
import numpy as np
"""
    This file includes unit test for mdp_dp.py
    You could test the correctness of your code by 
    typing 'nosetests -v mdp_dp_test.py' in the terminal
"""
env = gym.make("FrozenLake-v0")
env = env.unwrapped

#---------------------------------------------------------------
def test_python_version():
    '''------Dynamic Programming for MDP (100 points in total)------'''
    assert sys.version_info[0] == 3 # require python 2

#---------------------------------------------------------------
def test_policy_evaluation():
    '''policy_evaluation (20 points)'''
    random_policy1 = np.ones([env.nS, env.nA]) / env.nA
    V1 = policy_evaluation(env.P,env.nS,env.nA, random_policy1)
    test_v1 = np.array([0.003, 0.003, 0.009, 0.004, 0.006, 0., 0.026, 0., 0.018,
       0.057, 0.107, 0., 0., 0.13 , 0.391, 0.])

    np.random.seed(595)
    random_policy2 = np.random.rand(env.nS, env.nA)
    random_policy2 = random_policy2/random_policy2.sum(axis=1)[:,None]
    V2 = policy_evaluation(env.P,env.nS,env.nA, random_policy2)
    test_v2 = np.array([0.004, 0.006, 0.015, 0.006, 0.008, 0. , 0.042, 0. , 0.028,
       0.093, 0.173, 0. , 0. , 0.214, 0.503, 0. ])

    assert np.allclose(test_v1,V1,atol=1e-3)
    assert np.allclose(test_v2,V2,atol=1e-3)

#---------------------------------------------------------------
def test_policy_improvement():
    '''policy_improvement (20 points)'''
    np.random.seed(595)
    V1 = np.random.rand(env.nS)
    new_policy1 = policy_improvement(env.P, env.nS, env.nA, V1)
    test_policy1 = np.array([[1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 0., 1., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 1., 0.],
       [1., 0., 0., 0.]])

    V2 = np.zeros(env.nS)
    new_policy2 = policy_improvement(env.P, env.nS, env.nA, V2)
    test_policy2 = np.array([[1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.]])
    
    assert np.allclose(test_policy1,new_policy1)
    assert np.allclose(test_policy2,new_policy2)

    
#---------------------------------------------------------------
def test_policy_iteration():
    '''policy_iteration (20 points)'''
    random_policy1 = np.ones([env.nS, env.nA]) / env.nA

    np.random.seed(595)
    random_policy2 = np.random.rand(env.nS, env.nA)
    random_policy2 = random_policy2/random_policy2.sum(axis=1)[:,None]

    policy_pi1, V_pi1 = policy_iteration(env.P, env.nS, env.nA, random_policy1)
    policy_pi2, V_pi2 = policy_iteration(env.P, env.nS, env.nA, random_policy2)

    optimal_policy = np.array([[1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 1., 0.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.]])
    
    optimal_V = np.array([0.065, 0.058, 0.073, 0.054, 0.089, 0., 0.111, 0., 0.143,
       0.246, 0.299, 0., 0., 0.379, 0.639, 0.])

    assert np.allclose(policy_pi1,optimal_policy)
    assert np.allclose(V_pi1,optimal_V,atol=1e-2)
    assert np.allclose(policy_pi2,optimal_policy)
    assert np.allclose(V_pi2,optimal_V,atol=1e-2)

#---------------------------------------------------------------
def test_value_iteration():
    '''value_iteration (20 points)'''
    np.random.seed(10000)
    V1 = np.random.rand(env.nS)
    policy_vi1, V_vi1 = value_iteration(env.P, env.nS, env.nA, V1)

    V2 = np.zeros(env.nS)
    policy_vi2, V_vi2 = value_iteration(env.P, env.nS, env.nA, V2)

    optimal_policy = np.array([[1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 1., 0.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.]])
    
    optimal_V = np.array([0.065, 0.058, 0.073, 0.054, 0.089, 0., 0.111, 0., 0.143,
       0.246, 0.299, 0., 0., 0.379, 0.639, 0.])
    
    assert np.allclose(policy_vi1,optimal_policy)
    assert np.allclose(V_vi1,optimal_V,atol=1e-2)
    assert np.allclose(policy_vi2,optimal_policy)
    assert np.allclose(V_vi2,optimal_V,atol=1e-2)

#---------------------------------------------------------------            
def test_render_single():
    '''render_single (20 points)'''                 
    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    p_pi, V_pi = policy_iteration(env.P, env.nS, env.nA, random_policy)
    r_pi = render_single(env, p_pi, False, 50)
    print("total rewards of PI: ",r_pi)
    
    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
    V = np.zeros(env.nS)
    p_vi, V_vi = value_iteration(env.P, env.nS, env.nA, V)
    r_vi = render_single(env, p_vi, False, 50)
    print("total rewards of VI: ",r_vi)
    
    assert r_pi > 35
    assert r_vi > 35