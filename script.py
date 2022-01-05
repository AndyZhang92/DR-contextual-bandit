import numpy as np
import matplotlib.pyplot as plt
from config import Configuration
from linearPolicy import linearPolicyLearner
from data import dataGeneratorLinear, dataGenerator
from numba import jit
import pickle
from tqdm import trange

def adversarial_reward_experiment(n_train = 1000, n_test = 20000, n_exp = 100, delta = 0.1, reg_param = 0.01):
    """
    def compute_reward(theta, features_mat, rewards_mat):
        num = features_mat.shape[0]
        theta_act = np.argmax(features_mat @ theta, axis = 1)
        return np.mean(rewards_mat[np.arange(num), theta_act])
    """
    def compute_worst_bootstrap_reward(theta, features_mat, rewards_mat, repeat_num = 100):
        temp_res = []
        for i in range(repeat_num):
            num = features_mat.shape[0]
            boot_idx = np.random.randint(num, size = num)
            theta_act = np.argmax(features_mat[boot_idx] @ theta, axis = 1)
            temp_res = temp_res + [np.mean(rewards_mat[boot_idx, theta_act])]
        return np.min(temp_res)
    
    config = Configuration(mode = "linear")
    config.noise_level, config.p, config.f1, config.f2 = 2,10,0,1
    data_generator = dataGeneratorLinear(config)
    
    data_test = data_generator.generate_all_data(n_test)
    features_test_mat = data_test[0]
    exp_rewards_test_mat = data_test[3]
    rewards_test_mat = data_test[4]
    
    reward_LR = np.zeros(n_exp)
    reward_logistic = np.zeros(n_exp)
    reward_logistic_stable = np.zeros(n_exp)
    reward_DR = np.zeros(n_exp)
    reward_DR_ind = np.zeros(n_exp)
    reward_DR_ind_stable = np.zeros(n_exp)
    
    for i in trange(n_exp):
        (features_mat, opt_actions, actions, 
         exp_rewards_mat, rewards_mat, pi_mat, 
         IPW_mat, AIPW_mat) = data_generator.generate_all_data(n_train)
        
        theta_LR = linearPolicyLearner().linear_regression_policy(features_mat, rewards_mat, actions)
        theta_logistic = linearPolicyLearner().logistic_policy(features_mat, IPW_mat, reg_param = reg_param)
        theta_logistic_stable = linearPolicyLearner().logistic_policy_stable(
            features_mat, rewards_mat, actions, pi_mat, max_iter = 1000, reg_param = reg_param)
        DR_theta_logistic = linearPolicyLearner().DR_logistic_policy(
            features_mat, rewards_mat, delta = delta, reg_param = reg_param, verbose = False)
        DR_ind_theta_logistic = linearPolicyLearner().DR_ind_logistic_policy(
            features_mat, rewards_mat, actions, pi_mat, delta = delta, reg_param = reg_param, verbose = False)
        DR_ind_theta_logistic_stable = linearPolicyLearner().DR_ind_logistic_policy_stable(
            features_mat, rewards_mat, actions, pi_mat, delta = delta, reg_param = reg_param, verbose = False)
        
        reward_LR[i] = compute_worst_bootstrap_reward(theta_LR, features_test_mat, rewards_test_mat)
        reward_logistic[i] = compute_worst_bootstrap_reward(theta_logistic, features_test_mat, rewards_test_mat)
        reward_logistic_stable[i] = compute_worst_bootstrap_reward(theta_logistic_stable, features_test_mat, rewards_test_mat)
        reward_DR[i] = compute_worst_bootstrap_reward(DR_theta_logistic, features_test_mat, rewards_test_mat)
        reward_DR_ind[i] = compute_worst_bootstrap_reward(DR_ind_theta_logistic, features_test_mat, rewards_test_mat)
        reward_DR_ind_stable[i] = compute_worst_bootstrap_reward(DR_ind_theta_logistic_stable, features_test_mat, rewards_test_mat)
    
    reward_res = (reward_LR, reward_logistic, reward_logistic_stable, reward_DR, reward_DR_ind, reward_DR_ind_stable)
    file = open("./result/linear/bootstrap_ntrain_{}_delta_{}_reg_{}".format(n_train, delta, reg_param), 'wb')
    pickle.dump(reward_res, file)
    file.close()

if __name__ == '__main__':
    for n_train in [100,250,500,1000,1500,2000,2500]:
        adversarial_reward_experiment(n_train = n_train, n_exp = 200)
    