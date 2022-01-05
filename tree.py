import numpy as np
class treeNode(object):
    
    def __init__(self):
        self.left = None
        self.right = None
        self.split_dim = None
        self.split_val = None
        self.action = None
        
    def __str__(self):
        def _list_str(self):
            res_str = []
            if self.left:
                res_str = res_str + [" "*7+"<"+" "*7 + _str for _str in _list_str(self.left)]
            if self.split_val is not None:
                res_str += ["dim: {0}, split: {1:f}".format(self.split_dim, self.split_val)]
            else:
                res_str += ["action: {}".format(self.action)]
            if self.right:
                res_str = res_str + [" "*7+">"+" "*7 + _str for _str in _list_str(self.right)]
            return res_str
        return "\n".join(_list_str(self))
    
    def eval_tree(self, feature, reward_mat):
        if feature.size == 0:
            raise Exception('feature.size = 0')
        if self.action is not None:
            return np.mean(reward_mat[:,self.action])
        else:
            idx_left = (feature[:, self.split_dim] <= self.split_val)
            idx_right = (feature[:, self.split_dim] > self.split_val)
            
            ans = self.left.eval_tree(feature[idx_left,:], reward_mat[idx_left,:]) * np.sum(idx_left)
            ans +=  self.right.eval_tree(feature[idx_right,:], reward_mat[idx_right,:]) * np.sum(idx_right)
            return ans / feature.shape[0]
        
    def classify(self, feature):
        num = feature.shape[0]
        return_action = np.zeros(num)
        if self.action is not None:
            return_action[:] = self.action
            return return_action
        idx_left = (feature[:, self.split_dim] <= self.split_val)
        idx_right = (feature[:, self.split_dim] > self.split_val)
        return_action[idx_left] = self.left.classify(feature[idx_left,:])
        return_action[idx_right] = self.right.classify(feature[idx_right,:])   
        return return_action

class treeLearner(object):
    
    def __init__(self):
        pass
    
    def _get_feature_order_mat(self, features):
        num, p = features.shape
        order_mat = np.stack([sorted(range(num), key = lambda x: features[x,i]) for i in range(p)], axis = 1)
        return order_mat.astype(int)
    
    def _get_actions_mat(self, actions, k, pi_mat):
        return np.eye(k)[actions.astype(int)] / pi_mat
        
    def find_split(self, features, rewards_mat):
        num, p = features.shape
        k = rewards_mat.shape[1]
        order_mat = self._get_feature_order_mat(features)
        equal_val_mat = np.column_stack([features[order_mat[1:, fdim],fdim] 
                                 == features[order_mat[:-1, fdim],fdim] 
                                 for fdim in range(features.shape[1])])
        total_reward_mat = [] # Total reward matrix for different split choices
        
        # Compute reward for different choices of split
        for fdim in range(p):
            cum_sum_group1 = np.cumsum(rewards_mat[order_mat[:,fdim],:],axis = 0)
            cum_sum_group1 = np.row_stack([np.zeros((1,k)), cum_sum_group1])
            cum_sum_group2 = np.flip(np.cumsum(rewards_mat[np.flip(order_mat[:,fdim]),:],axis = 0))
            cum_sum_group2 = np.row_stack([cum_sum_group2, np.zeros((1,k))])
            total_reward_mat += [np.max(cum_sum_group1,axis = 1) + np.max(cum_sum_group2,axis = 1)]

        total_reward_mat = np.column_stack(total_reward_mat)
        total_reward_mat[1:-1,:][equal_val_mat] = np.NINF
        # find the optimal split

        arg_max = np.argmax(total_reward_mat)
        opt_dim, opt_split_order = arg_max % p, arg_max // p
        
        group1_id = order_mat[:opt_split_order, opt_dim]
        group2_id = order_mat[opt_split_order:, opt_dim]
        
        if opt_split_order == 0:
            opt_split = np.NINF
        elif opt_split_order == num:
            opt_split = np.inf
        else:
            opt_split = (features[order_mat[opt_split_order-1,opt_dim], opt_dim]
                         + features[order_mat[opt_split_order,opt_dim], opt_dim]) / 2
        #print(np.max(features[group1_id,opt_dim]))
        return opt_dim, opt_split, group1_id, group2_id
    
    def find_split_with_indicator(self, features, rewards_mat, actions_mat):
        num, p = features.shape
        k = rewards_mat.shape[1]
        order_mat = self._get_feature_order_mat(features)
        equal_val_mat = np.column_stack([features[order_mat[1:, fdim],fdim] 
                                 == features[order_mat[:-1, fdim],fdim] 
                                 for fdim in range(features.shape[1])])

        total_reward_mat = [] # Total reward matrix for different split choices
        arg_max_reward_mat = []

        
        # Compute reward for different choices of split
        for fdim in range(p):
            cum_sum_group1 = np.cumsum(rewards_mat[order_mat[:,fdim],:],axis = 0)
            cum_sum_group1 = np.row_stack([np.zeros((1,k)), cum_sum_group1]) # add 0 padding
            cum_act_group1 = np.cumsum(actions_mat[order_mat[:,fdim],:],axis = 0)
            cum_act_group1 = np.row_stack([np.zeros((1,k)), cum_act_group1]) # add 0 padding
            
            cum_sum_group2 = np.flip(np.cumsum(rewards_mat[np.flip(order_mat[:,fdim]),:],axis = 0))
            cum_sum_group2 = np.row_stack([cum_sum_group2, np.zeros((1,k))]) # add 0 padding
            cum_act_group2 = np.flip(np.cumsum(actions_mat[np.flip(order_mat[:,fdim]),:],axis = 0))
            cum_act_group2 = np.row_stack([cum_act_group2, np.zeros((1,k))]) # add 0 padding
            
            kron_sum = np.kron(cum_sum_group1, np.ones((1,k))) + np.kron(np.ones((1,k)),cum_sum_group2)
            kron_act = np.kron(cum_act_group1, np.ones((1,k))) + np.kron(np.ones((1,k)),cum_act_group2)
            
            total_reward_mat += [np.max((kron_sum/kron_act), axis = 1)]
            
        total_reward_mat = np.column_stack(total_reward_mat)
        total_reward_mat[1:-1,:][equal_val_mat] = np.NINF

        
        arg_max = np.argmax(total_reward_mat)
        opt_dim, opt_split_order = arg_max % p, arg_max // p
        
        group1_id = order_mat[:opt_split_order, opt_dim]
        group2_id = order_mat[opt_split_order:, opt_dim]
        
        if opt_split_order == 0:
            opt_split = np.NINF
        elif opt_split_order == num:
            opt_split = np.inf
        else:
            opt_split = (features[order_mat[opt_split_order-1,opt_dim], opt_dim]
                         + features[order_mat[opt_split_order,opt_dim], opt_dim]) / 2
        #print(np.max(features[group1_id,opt_dim]))
        return opt_dim, opt_split, group1_id, group2_id
    
    def find_opt_act(self, rewards_mat):
        return np.argmax(np.mean(rewards_mat, axis = 0))
    
    def find_opt_act_with_indicator(self, rewards_mat, actions_mat):
        return np.argmax(np.mean(rewards_mat, axis = 0)/(np.mean(actions_mat, axis = 0)-1e-8))
    
    def learn_tree(self, features, rewards_mat, level_of_split = 1):
        #print(np.min(features[:,7]), np.max(features[:,7]), level_of_split)
        if level_of_split == 1:
            opt_dim, opt_split, group1_id, group2_id = self.find_split(features, rewards_mat)
            root = treeNode()
            root.split_dim = opt_dim
            root.split_val = opt_split
            left_node, right_node = treeNode(), treeNode()
            left_node.action = self.find_opt_act(rewards_mat[group1_id,:])
            right_node.action = self.find_opt_act(rewards_mat[group2_id,:])
            root.left = left_node
            root.right = right_node
        else:
            opt_dim, opt_split, group1_id, group2_id = self.find_split(features, rewards_mat)
            #print(np.max(features[group1_id,7]))
            root = treeNode()
            root.split_dim = opt_dim 
            root.split_val = opt_split
            root.left = self.learn_tree(features[group1_id,:], rewards_mat[group1_id,:],
                                        level_of_split = level_of_split - 1)
            root.right = self.learn_tree(features[group2_id,:], rewards_mat[group2_id,:],
                                        level_of_split = level_of_split - 1)
        return root
    
    def DR_tree_learner(self, features, rewards_mat, delta = 0.5, level_of_split = 1, 
                       max_iter = 50, verbose = True):
        alpha = 4/(10*delta)
        n = features.shape[0]
        tol = 1e-8
        last_reward = np.inf
        for i in range(max_iter):
            rewardmat_DR = -np.exp(-rewards_mat/alpha)
            tree = self.learn_tree(features, rewardmat_DR, level_of_split)
            tree_act = tree.classify(features)
            Y_pi = rewards_mat[range(n),tree_act.astype(int)]
            W_n = np.mean(np.exp(-Y_pi/alpha))
            d_W_n = np.mean(Y_pi *np.exp(-Y_pi/alpha)/ (alpha*alpha))
            d2_W_n = np.mean(np.exp(-Y_pi/alpha) * ((Y_pi**2)/(alpha**4)-(2*Y_pi)/(alpha**3)))
            DR_reward = -alpha*np.log(W_n)-alpha*delta
            d_DR_reward = -np.log(W_n) - delta - alpha*d_W_n/W_n
            d2_DR_reward = -2*d_W_n/W_n - alpha * (d2_W_n*W_n - d_W_n**2)/(W_n**2)
            alpha = alpha - d_DR_reward/d2_DR_reward
            if verbose:
                print("iter={:d},  alpha={:.4f}, DR_reward = {:.4f}".format(i,alpha,DR_reward))
            if (abs(last_reward - DR_reward) < tol):
                if verbose:
                    print('Early break...')
                break
            last_reward = DR_reward
        return tree

    def DR_tree_learner_partial_info(self, features, rewards_mat, actions_mat, delta = 0.5, level_of_split = 1, 
    	max_iter = 50, alpha_init = None, verbose = True):
        if alpha_init == None:
            alpha = 4/(10*delta)
        else:
            alpha = alpha_init
        n = features.shape[0]
        tol = 1e-8
        last_reward = np.inf
        for i in range(max_iter):
            rewardmat_DR = (-np.exp(-rewards_mat/alpha)) * actions_mat
            tree = self.learn_tree(features, rewardmat_DR, level_of_split)
            tree_act = tree.classify(features)
            Y_pi = rewards_mat[range(n),tree_act.astype(int)]
            W_n = np.mean(np.exp(-Y_pi/alpha))
            d_W_n = np.mean(Y_pi *np.exp(-Y_pi/alpha)/ (alpha*alpha))
            d2_W_n = np.mean(np.exp(-Y_pi/alpha) * ((Y_pi**2)/(alpha**4)-(2*Y_pi)/(alpha**3)))
            DR_reward = -alpha*np.log(W_n)-alpha*delta
            d_DR_reward = -np.log(W_n) - delta - alpha*d_W_n/W_n
            d2_DR_reward = -2*d_W_n/W_n - alpha * (d2_W_n*W_n - d_W_n**2)/(W_n**2)
            alpha = alpha - d_DR_reward/d2_DR_reward
            if verbose:
                print("iter={:d},  alpha={:.4f}, DR_reward = {:.4f}".format(i,alpha,DR_reward))
            if (abs(last_reward - DR_reward) < tol):
                if verbose:
                    print('Early break...')
                break
            last_reward = DR_reward
        return tree

    def learn_tree_with_indicator(self, features, rewards_mat, actions_mat, level_of_split = 1):
        #print(np.min(features[:,7]), np.max(features[:,7]), level_of_split)
        if level_of_split == 1:
            opt_dim, opt_split, group1_id, group2_id = self.find_split_with_indicator(features, rewards_mat, actions_mat)
            root = treeNode()
            root.split_dim = opt_dim
            root.split_val = opt_split
            left_node, right_node = treeNode(), treeNode()
            left_node.action = self.find_opt_act_with_indicator(rewards_mat[group1_id,:],
                                                               actions_mat[group1_id,:])
            right_node.action = self.find_opt_act_with_indicator(rewards_mat[group2_id,:],
                                                                actions_mat[group2_id,:])
            root.left = left_node
            root.right = right_node
        else:
            opt_dim, opt_split, group1_id, group2_id = self.find_split_with_indicator(features, rewards_mat, actions_mat)
            #print(np.max(features[group1_id,7]))
            root = treeNode()
            root.split_dim = opt_dim 
            root.split_val = opt_split
            root.left = self.learn_tree_with_indicator(features[group1_id,:], rewards_mat[group1_id,:],
                                        actions_mat[group1_id,:],level_of_split = level_of_split - 1)
            root.right = self.learn_tree_with_indicator(features[group2_id,:], rewards_mat[group2_id,:],
                                        actions_mat[group2_id,:], level_of_split = level_of_split - 1)
        return root
    
    def DR_tree_learner_with_indicator(self, features, rewards_mat, actions_mat, delta = 0.5, level_of_split = 1, 
                       max_iter = 50, alpha_init = None, verbose = True):
        if alpha_init == None:
            alpha = 4/(10*delta)
        else:
            alpha = alpha_init
        n = features.shape[0]
        tol = 1e-8
        last_reward = np.inf
        for i in range(max_iter):
            rewardmat_DR = (-np.exp(-rewards_mat/alpha)) * actions_mat
            tree = self.learn_tree_with_indicator(features, rewardmat_DR, actions_mat, level_of_split)
            tree_act = tree.classify(features)
            Y_pi = rewards_mat[range(n),tree_act.astype(int)]
            W_n = np.mean(np.exp(-Y_pi/alpha))
            d_W_n = np.mean(Y_pi * np.exp(-Y_pi/alpha)/ (alpha*alpha))
            d2_W_n = np.mean(np.exp(-Y_pi/alpha) * ((Y_pi**2)/(alpha**4)-(2*Y_pi)/(alpha**3)))
            DR_reward = -alpha*np.log(W_n)-alpha*delta
            d_DR_reward = -np.log(W_n) - delta - alpha*d_W_n/W_n
            d2_DR_reward = -2*d_W_n/W_n - alpha * (d2_W_n*W_n - d_W_n**2)/(W_n**2)
            if d_DR_reward/d2_DR_reward > 0 and d_DR_reward/d2_DR_reward > alpha:
                alpha = alpha/2
            else:
                alpha = alpha - d_DR_reward/d2_DR_reward
            if verbose:
                print("iter={:d},  alpha={:.4f}, DR_reward = {:.4f}".format(i,alpha,DR_reward))
            if (abs(last_reward - DR_reward) < tol):
                if verbose:
                    print('Early break...')
                break
            last_reward = DR_reward
        return tree
    
    def DR_eval_tree(self, tree, features, rewards_mat, delta = 0.5, weights = None, max_iter = 50, 
                     verbose = True, alpha_init = None, return_alpha = False, fixed_alpha = None):
        if alpha_init == None:
            alpha = 2
        else:
            alpha = alpha_init
        n = features.shape[0]
        tol = 1e-8
        last_reward = np.inf
        if weights is None:
            weights = np.ones(n) / n
        if fixed_alpha is not None:
            alpha = fixed_alpha
            # rewardmat_DR = -np.exp(-rewards_mat/alpha) 
            tree_act = tree.classify(features)
            Y_pi = rewards_mat[range(n),tree_act.astype(int)]
            W_n = np.sum(weights * np.exp(-Y_pi/alpha))
            DR_reward = -alpha*np.log(W_n)-alpha*delta
            return DR_reward
        if delta < 1e-8:
            tree_act = tree.classify(features)
            return np.sum(weights * rewards_mat[range(n),tree_act.astype(int)])
        for i in range(max_iter):
            rewardmat_DR = -np.exp(-rewards_mat/alpha) 
            tree_act = tree.classify(features)
            Y_pi = rewards_mat[range(n),tree_act.astype(int)]
            W_n = np.sum(weights * np.exp(-Y_pi/alpha))
            d_W_n = np.sum(weights * Y_pi *np.exp(-Y_pi/alpha)/ (alpha*alpha))
            d2_W_n = np.sum(weights * np.exp(-Y_pi/alpha) * ((Y_pi**2)/(alpha**4)-(2*Y_pi)/(alpha**3)))
            DR_reward = -alpha*np.log(W_n)-alpha*delta
            d_DR_reward = -np.log(W_n) - delta - alpha*d_W_n/W_n
            d2_DR_reward = -2*d_W_n/W_n - alpha * (d2_W_n*W_n - d_W_n**2)/(W_n**2)
            alpha = alpha - d_DR_reward/d2_DR_reward
            if verbose:
                print("iter={:d},  alpha={:.4f}, DR_reward = {:.4f}".format(i,alpha,DR_reward))
            if (abs(last_reward - DR_reward) < tol):
                if verbose:
                    print('Early break...')
                break
            last_reward = DR_reward
        if return_alpha:
        	return last_reward, alpha
        else:
        	return last_reward

    def DR_eval_tree_partial_info(self, tree, features, rewards_mat, actions_mat, delta = 0.5, 
        weights = None, max_iter = 50, verbose = True):
        alpha = 2
        n = features.shape[0]
        tol = 1e-8
        last_reward = np.inf
        if weights is None:
            weights = np.ones(n) / n
        if delta < 1e-8:
            tree_act = tree.classify(features)
            return np.sum(weights *(rewards_mat*actions_mat)[range(n),tree_act.astype(int)]) #/ np.mean(actions == tree_act)
        for i in range(max_iter):
            rewardmat_DR = -np.exp(-rewards_mat/alpha) * actions_mat
            tree_act = tree.classify(features)
            Y_pi = rewards_mat[range(n),tree_act.astype(int)]
            I_i = actions_mat[range(n), tree_act.astype(int)]
            W_n = np.average(np.exp(-Y_pi/alpha) * I_i, weights = weights)
            d_W_n = np.average(Y_pi *np.exp(-Y_pi/alpha)* I_i / (alpha*alpha), weights = weights)
            d2_W_n = np.average(np.exp(-Y_pi/alpha) *I_i*((Y_pi**2) /(alpha**4)-(2*Y_pi)/(alpha**3)), weights = weights)
            DR_reward = -alpha*np.log(W_n)-alpha*delta
            d_DR_reward = -np.log(W_n) - delta - alpha*d_W_n/W_n
            d2_DR_reward = -2*d_W_n/W_n - alpha * (d2_W_n*W_n - d_W_n**2)/(W_n**2)
            alpha = alpha - d_DR_reward/d2_DR_reward
            if verbose:
                print("iter={:d},  alpha={:.4f}, DR_reward = {:.4f}".format(i,alpha,DR_reward))
            if (abs(last_reward - DR_reward) < tol):
                if verbose:
                    print('Early break...')
                break
            last_reward = DR_reward
        return last_reward

    def DR_eval_tree_partial_info_divide_ind(self, tree, features, rewards_mat, actions_mat, delta = 0.5, 
        weights = None, max_iter = 50, verbose = True, return_alpha = False):
        alpha = 2
        n = features.shape[0]
        tol = 1e-8
        last_reward = np.inf
        if weights is None:
            weights = np.ones(n) / n
        if delta < 1e-8:
            tree_act = tree.classify(features)
            I_i = actions_mat[range(n), tree_act.astype(int)]
            return np.sum(weights *(rewards_mat*actions_mat)[range(n),tree_act.astype(int)]) / np.sum(weights * I_i)
        for i in range(max_iter):
            rewardmat_DR = -np.exp(-rewards_mat/alpha) * actions_mat
            tree_act = tree.classify(features)
            Y_pi = rewards_mat[range(n),tree_act.astype(int)]
            I_i = actions_mat[range(n), tree_act.astype(int)]
            W_n = np.sum(weights * np.exp(-Y_pi/alpha) * I_i) / np.sum(weights * I_i)
            d_W_n = np.sum(weights * Y_pi *np.exp(-Y_pi/alpha)* I_i / (alpha*alpha)) / np.sum(weights * I_i)
            d2_W_n = np.sum(weights * np.exp(-Y_pi/alpha) *I_i*((Y_pi**2) / (alpha**4)-(2*Y_pi)/(alpha**3))) \
            / np.sum(weights * I_i)
            DR_reward = -alpha*np.log(W_n)-alpha*delta
            d_DR_reward = -np.log(W_n) - delta - alpha*d_W_n/W_n
            d2_DR_reward = -2*d_W_n/W_n - alpha * (d2_W_n*W_n - d_W_n**2)/(W_n**2)
            alpha = alpha - d_DR_reward/d2_DR_reward
            if verbose:
                print("iter={:d},  alpha={:.4f}, DR_reward = {:.4f}".format(i,alpha,DR_reward))
            if (abs(last_reward - DR_reward) < tol):
                if verbose:
                    print('Early break...')
                break
            last_reward = DR_reward
        if return_alpha:
        	return last_reward, alpha
        else:
        	return last_reward