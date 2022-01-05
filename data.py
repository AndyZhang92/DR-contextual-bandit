import numpy as np

class dataGenerator(object):
    # DATA GENERATION CLASS
    def __init__(self, config, mode = "train"):
        self.p = config.p
        self.k = config.k
        self.noise_level = config.noise_level
        self.f1 = config.f1
        self.f2 = config.f2

        if mode == "train":
        	self.shape = config.train_shape
        else:
        	self.shape = config.test_shape
        
    def generate_features(self, num):
        return np.random.uniform(size = (num, self.p))
    
    def generate_best_actions(self, features):
        f1, f2 = self.f1, self.f2
        num = features.shape[0]
        opt_actions = np.ones(num)
        idx = np.logical_and(features[:,f1] <= 0.6, features[:,f2] >= 0.35)
        opt_actions[idx] = 0
        idx = (np.square((features[:,f1]/self.shape['r11'])) 
               + np.square((features[:,f2]/self.shape['r12'])) <= 1)
        opt_actions[idx] = 2
        idx = (np.square((1-features[:,f1])/self.shape['r21'])
               + np.square((1-features[:,f2])/self.shape['r22']) <= 1)
        opt_actions[idx] = 2
        return opt_actions
        
    def generate_actions(self, features):
        num = features.shape[0]
        f1, f2 = self.f1, self.f2
        opt_actions = self.generate_best_actions(features)
        tmp = np.random.uniform(size = (num))
        actions = np.ones(num)
        id0 = np.logical_and(tmp <= 0.4, opt_actions == 2)
        id1 = np.logical_and(tmp > 0.6, opt_actions == 2)
        actions[id0], actions[id1] = 0, 2
        id0 = np.logical_and(tmp <= 0.2, opt_actions != 2)
        id1 = np.logical_and(tmp > 0.8, opt_actions != 2)
        actions[id0], actions[id1] = 0, 2
        return actions
    
    def generate_reward(self, features, actions):
        num = features.shape[0]
        opt_actions = self.generate_best_actions(features)
        noise = np.random.randn(num)
        rewards = np.zeros(num)
        """
        idx = (opt_actions == 2)
        rewards[idx] = 1.5 * (actions[idx] - 1) + self.noise_level * noise[idx]
        idx = (opt_actions == 1)
        rewards[idx] = 2 - np.abs(actions[idx] - 1)/2 + self.noise_level * noise[idx]
        idx = (opt_actions == 0)
        rewards[idx] = 3 - actions[idx] + self.noise_level * noise[idx]
        """
        idx = (opt_actions == 2)
        rewards[idx] = (actions[idx]) + self.noise_level * noise[idx]
        idx = (opt_actions == 1)
        rewards[idx] = - np.abs(actions[idx] - 1) + self.noise_level * noise[idx]
        idx = (opt_actions == 0)
        rewards[idx] = - actions[idx] + self.noise_level * noise[idx]
        return rewards

    def generate_exp_reward_mat(self, features):
        num = features.shape[0]
        opt_actions = self.generate_best_actions(features)
        exp_rewards = np.zeros((num,self.k))
        """
        for j in range(self.k):
            exp_rewards[opt_actions == 0, j] = 3 - j
        for j in range(self.k):
            exp_rewards[opt_actions == 1, j] = 2 - np.abs(j - 1)/2
        for j in range(self.k):
            exp_rewards[opt_actions == 2, j] = 1.5*(j - 1)
        """
        for j in range(self.k):
            exp_rewards[opt_actions == 0, j] = - j
        for j in range(self.k):
            exp_rewards[opt_actions == 1, j] = - np.abs(j - 1)
        for j in range(self.k):
            exp_rewards[opt_actions == 2, j] = j
        return exp_rewards
        
    def generate_reward_mat(self, features):
        num = features.shape[0]
        opt_actions = self.generate_best_actions(features)
        noise = np.random.randn(num, self.k)
        return (self.generate_exp_reward_mat(features) + self.noise_level * noise)
    
    def generate_pi_mat(self, features):
        num = features.shape[0]
        opt_actions = self.generate_best_actions(features)
        pi_mat = np.zeros((num, self.k))
        pi_mat[opt_actions == 2,:] += [0.4,0.2,0.4]
        pi_mat[opt_actions != 2,:] += [0.2,0.6,0.2]
        return pi_mat
    def generate_all_data(self, n_exp):
        features_mat = self.generate_features(n_exp)
        opt_actions = self.generate_best_actions(features_mat)
        actions = self.generate_actions(features_mat)
        exp_rewards_mat = self.generate_exp_reward_mat(features_mat)
        rewards_mat = self.generate_reward_mat(features_mat)
        pi_mat = self.generate_pi_mat(features_mat)

        IPW_mat = np.zeros_like(rewards_mat)
        IPW_mat[np.arange(n_exp),actions.astype(np.int)] = \
        (rewards_mat/pi_mat)[np.arange(n_exp),actions.astype(np.int)]

        AIPW_mat = np.zeros_like(rewards_mat)
        AIPW_mean = np.mean(IPW_mat, axis = 0)
        AIPW_mat = AIPW_mat + AIPW_mean
        AIPW_mat[np.arange(n_exp),actions.astype(np.int)] += \
        ((rewards_mat - AIPW_mean)/pi_mat)[np.arange(n_exp),actions.astype(np.int)]

        return (features_mat, opt_actions, actions, 
            exp_rewards_mat, rewards_mat, pi_mat,
            IPW_mat, AIPW_mat)

class dataGeneratorLinear(object):

    def __init__(self, config, mode = "train"):
        self.p = config.p
        self.k = config.k
        self.noise_level = config.noise_level
        self.f1 = config.f1
        self.f2 = config.f2

    def generate_features(self, num):
        return np.random.uniform(low = -1.0, high = 1.0, size = (num, self.p))

    def generate_best_actions(self, features):
        f1, f2 = self.f1, self.f2
        num = features.shape[0]
        opt_actions = np.ones(num)

        idx = np.logical_and(np.sqrt(3)*features[:,f1] <= features[:,f2], features[:,f2] > 0)
        opt_actions[idx] = 0
        idx = np.logical_and(np.sqrt(3)*features[:,f1] <= -features[:,f2], features[:,f2] <= 0)
        opt_actions[idx] = 2
        return opt_actions

    def generate_actions(self, features):
        num = features.shape[0]
        f1, f2 = self.f1, self.f2
        opt_actions = self.generate_best_actions(features)
        tmp = np.random.uniform(size = (num))
        actions = np.ones(num)
        id0 = np.logical_and(tmp <= 0.4, opt_actions == 2)
        id1 = np.logical_and(tmp > 0.6, opt_actions == 2)
        actions[id0], actions[id1] = 0, 2
        id0 = np.logical_and(tmp <= 0.2, opt_actions != 2)
        id1 = np.logical_and(tmp > 0.8, opt_actions != 2)
        actions[id0], actions[id1] = 0, 2
        return actions

    def generate_exp_reward_mat(self, features):
        num = features.shape[0]
        opt_actions = self.generate_best_actions(features)
        exp_rewards = np.zeros((num,self.k))
        """
        for j in range(self.k):
            exp_rewards[opt_actions == 0, j] = 3 - j
        for j in range(self.k):
            exp_rewards[opt_actions == 1, j] = 2 - np.abs(j - 1)/2
        for j in range(self.k):
            exp_rewards[opt_actions == 2, j] = 1.5*(j - 1)
        """
        for j in range(self.k):
            exp_rewards[opt_actions == 0, j] = 3 - j
        for j in range(self.k):
            exp_rewards[opt_actions == 1, j] = 2 - np.abs(j - 1) / 2
        for j in range(self.k):
            exp_rewards[opt_actions == 2, j] = 1.5 * (j-1)
        return exp_rewards

    def generate_reward_mat(self, features):
        num = features.shape[0]
        opt_actions = self.generate_best_actions(features)
        noise = np.random.randn(num, self.k)
        return (self.generate_exp_reward_mat(features) + self.noise_level * noise)
    
    def generate_pi_mat(self, features):
        num = features.shape[0]
        opt_actions = self.generate_best_actions(features)
        pi_mat = np.zeros((num, self.k))
        pi_mat[opt_actions == 2,:] += [0.4,0.2,0.4]
        pi_mat[opt_actions != 2,:] += [0.2,0.6,0.2]
        return pi_mat

    def generate_all_data(self, n_exp):
        features_mat = self.generate_features(n_exp)
        opt_actions = self.generate_best_actions(features_mat)
        actions = self.generate_actions(features_mat)
        exp_rewards_mat = self.generate_exp_reward_mat(features_mat)
        rewards_mat = self.generate_reward_mat(features_mat)
        pi_mat = self.generate_pi_mat(features_mat)

        IPW_mat = np.zeros_like(rewards_mat)
        IPW_mat[np.arange(n_exp),actions.astype(np.int)] = \
        (rewards_mat/pi_mat)[np.arange(n_exp),actions.astype(np.int)]

        AIPW_mat = np.zeros_like(rewards_mat)
        AIPW_mean = np.mean(IPW_mat, axis = 0)
        AIPW_mat = AIPW_mat + AIPW_mean
        AIPW_mat[np.arange(n_exp),actions.astype(np.int)] += \
        ((rewards_mat - AIPW_mean)/pi_mat)[np.arange(n_exp),actions.astype(np.int)]

        return (features_mat, opt_actions, actions, 
            exp_rewards_mat, rewards_mat, pi_mat,
            IPW_mat, AIPW_mat)