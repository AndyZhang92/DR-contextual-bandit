import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from data import dataGenerator
from tree import treeNode, treeLearner
import pickle

class Configuration(object):
    # EXPERIMENT CONFIGURATION
    def __init__(self):
        self.n_best = 50000 # number of observations to calculate best tree
        self.n_test = 40000 # number of observations to test learned trees
        self.p = 10 # number of features
        
        self.f1 = 5 # first essential feature
        self.f2 = 7 # second essential feature
        
        self.k = 3 # number of actions
        self.noise_level = 2
        self.delta = 0.05 #size of uncertainty ball
        
        self.num_exp = 1000
        self.n_exp_list = [1000,1500,2000,3000,4000,5000,7000,10000,20000]

def regret_exp(tree_best, n_exp, config):
    # Generate training data
    np.random.seed()
    features_train = data_generator.generate_features(n_exp)
    actions_train = data_generator.generate_actions(features_train)
    rewards_mat_train = data_generator.generate_reward_mat(features_train)
    pi_mat_train = data_generator.generate_pi_mat(features_train)
    actions_mat_train = treeLearner()._get_actions_mat(actions_train, config.k, pi_mat_train)
    #ob_rewards_mat_train = 0*rewards_mat_train
    #ob_rewards_mat_train[range(n_exp),actions_train.astype(np.int)] \
    #= (rewards_mat_train/pi_mat_train)[range(n_exp),actions_train.astype(np.int)]

    # Train DRO tree
    tree_DRO = treeLearner().DR_tree_learner_with_indicator(features_train, rewards_mat_train,
                                                        actions_mat_train, level_of_split=2, 
                                                        delta = config.delta, verbose=False)
    features_test = data_generator.generate_features(config.n_test)
    rewards_mat_test = data_generator.generate_reward_mat(features_test)
    tree_OPT_reward = treeLearner().DR_eval_tree(tree_best, features_test, rewards_mat_test, delta = config.delta, verbose=False)
    tree_DRO_reward = treeLearner().DR_eval_tree(tree_DRO, features_test, rewards_mat_test, delta = config.delta, verbose=False)
    regret = tree_OPT_reward - tree_DRO_reward
    #print("OPT_tree_DRO_reward: {:4f}, tree_DRO_reward: {:4f}, regret: {:4f}".format(tree_OPT_reward, 
    #                                                                    tree_DRO_reward, 
    #                                                                    regret))
    return {"tree_DRO": tree_DRO,
            "OPT_reward": tree_OPT_reward, 
            "tree_DRO_reward": tree_DRO_reward, 
            "regret": regret}

if __name__ == "__main__":
	config = Configuration()
	data_generator = dataGenerator(config)
	features_best = data_generator.generate_features(config.n_best)
	exp_rewards_mat_best = data_generator.generate_exp_reward_mat(features_best)
	tree_best = treeLearner().learn_tree(features_best, exp_rewards_mat_best,level_of_split = 2)
	num_experiment = config.num_exp
	for num_train in config.n_exp_list:
	    print("delta = {}  train_num = {}".format(config.delta,num_train))
	    cores = cpu_count()
	    p = Pool(cores)
	    pbar = tqdm(total = num_experiment, leave = False)
	    result = []
	    def update_pbar(*a):
	        pbar.update()
	    for i in range(num_experiment):
	        result.append(p.apply_async(regret_exp, args=(tree_best, num_train, config), callback = update_pbar))
	    p.close()
	    p.join()
	    pbar.close()
	    result = [res.get() for res in result]
	    file_name = "./result/res_delta_{:3f}_train_{}".format(config.delta,num_train)
	    with open(file_name, mode = 'wb') as file:
	        pickle.dump(result, file)
