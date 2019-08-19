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
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    V = policy_evaluation(env.P,env.nS,env.nA, random_policy)
    test_v = np.array([0.003, 0.003, 0.009, 0.004, 0.006, 0., 0.026, 0., 0.018,
       0.057, 0.107, 0., 0., 0.13 , 0.391, 0.])
    assert np.allclose(test_v,V,atol=1e-3)

#---------------------------------------------------------------
def test_policy_improvement():
    '''policy_improvement (20 points)'''
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    V = policy_evaluation(env.P,env.nS,env.nA, random_policy)
    new_policy = policy_improvement(env.P, env.nS, env.nA, V)
    test_policy = np.array([[0.  , 0.5 , 0.5 , 0.  ],
       [0.  , 0.  , 0.  , 1.  ],
       [0.  , 0.  , 1.  , 0.  ],
       [0.  , 0.  , 0.  , 1.  ],
       [1.  , 0.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.5 , 0.  , 0.5 , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.  , 0.  , 0.  , 1.  ],
       [0.  , 1.  , 0.  , 0.  ],
       [1.  , 0.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.25, 0.25, 0.25, 0.25],
       [0.  , 0.  , 1.  , 0.  ],
       [0.  , 1.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25]])
    
    assert np.allclose(test_policy,new_policy)
    
#---------------------------------------------------------------
def test_policy_iteration():
    '''policy_iteration (20 points)'''
    policy_pi, V_pi = policy_iteration(env.P, env.nS, env.nA)
    optimal_policy = np.array([[1.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 1.  ],
       [1.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 1.  ],
       [1.  , 0.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.5 , 0.  , 0.5 , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.  , 0.  , 0.  , 1.  ],
       [0.  , 1.  , 0.  , 0.  ],
       [1.  , 0.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.25, 0.25, 0.25, 0.25],
       [0.  , 0.  , 1.  , 0.  ],
       [0.  , 1.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25]])
    
    optimal_V = np.array([0.065, 0.058, 0.073, 0.054, 0.089, 0., 0.111, 0., 0.143,
       0.246, 0.299, 0., 0., 0.379, 0.639, 0.])

    assert np.allclose(policy_pi,optimal_policy)
    assert np.allclose(V_pi,optimal_V,atol=1e-3)

#---------------------------------------------------------------
def test_value_iteration():
    '''value_iteration (20 points)'''
    policy_vi, V_vi = value_iteration(env.P, env.nS, env.nA)
    optimal_policy = np.array([[1.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 1.  ],
       [1.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 1.  ],
       [1.  , 0.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.5 , 0.  , 0.5 , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.  , 0.  , 0.  , 1.  ],
       [0.  , 1.  , 0.  , 0.  ],
       [1.  , 0.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25],
       [0.25, 0.25, 0.25, 0.25],
       [0.  , 0.  , 1.  , 0.  ],
       [0.  , 1.  , 0.  , 0.  ],
       [0.25, 0.25, 0.25, 0.25]])
    
    optimal_V = np.array([0.065, 0.058, 0.073, 0.054, 0.089, 0., 0.111, 0., 0.143,
       0.246, 0.299, 0., 0., 0.379, 0.639, 0.])
    
    assert np.allclose(policy_vi,optimal_policy)
    assert np.allclose(V_vi,optimal_V,atol=1e-3)

#---------------------------------------------------------------            
def test_render_single():
    '''render_single (20 points)'''                 
    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
    p_pi, V_pi = policy_iteration(env.P, env.nS, env.nA)
    r_pi = render_single(env, p_pi, False, 50)
    print("total rewards of PI: ",r_pi)
    
    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
    p_vi, V_vi = value_iteration(env.P, env.nS, env.nA)
    r_vi = render_single(env, p_vi, False, 50)
    print("total rewards of VI: ",r_vi)
    
    assert r_pi > 35
    assert r_vi > 35