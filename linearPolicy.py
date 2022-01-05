import numpy as np
from sklearn.linear_model import LinearRegression

class linearPolicyLearner(object):
    def __init__(self):
        pass

    def convex_relaxation(self, features_mat, rewards_mat, learning_rate = 0.1, max_iter = 10000):
        n = features_mat.shape[0]
        p = features_mat.shape[1]
        k = rewards_mat.shape[1]
        #theta = np.random.randn(p, k)
        theta = np.zeros((p, k))
        act_star_array = np.zeros(n)
        for i in range(n): 
            # numba does not support axis argument in np.argmax
            act_star_array[i] = np.argmax(rewards_mat[i,:])
        for i in range(1,max_iter+1):
            idx = np.random.randint(n)
            act_star = np.int64(act_star_array[idx])
            act_hat = np.argmin(rewards_mat[idx,:] - (features_mat[idx,:]@theta))
            theta[:, act_hat] = theta[:, act_hat] - (learning_rate)*features_mat[idx,:]
            theta[:, act_star] = theta[:, act_star] + (learning_rate)*features_mat[idx,:]
        return theta

    def logistic_policy(self, features_mat, rewards_mat, max_iter = 1000, 
        learning_rate = 1, reg_param = 0, theta_init = None, intercept = True):
        n = features_mat.shape[0]
        p = features_mat.shape[1]
        k = rewards_mat.shape[1]

        if theta_init is None:
            theta = np.zeros((p, k))
        else:
            theta = theta_init
        for i in range(max_iter):
            scores_mat = features_mat@theta
            scores_mat = scores_mat
            exp_scores_mat = np.exp(scores_mat)
            exp_scores_mat_div_sum_exp = ((1/np.sum(exp_scores_mat, axis = 1))*exp_scores_mat.T).T
            temp_mat = (exp_scores_mat_div_sum_exp - exp_scores_mat_div_sum_exp ** 2)
            temp_mat = temp_mat * rewards_mat
            grad_theta_mat = np.average((temp_mat.reshape((n,1,k)) * features_mat.reshape((n,p,1))),axis = 0)
            theta = theta + (learning_rate)*(grad_theta_mat - reg_param * theta)
            theta = (theta.T - theta[:, 0]).T
        #print(exp_scores_mat_div_sum_exp)
        return theta

    def logistic_policy_2(self, features_mat, rewards_mat, max_iter = 1000, 
        learning_rate = 1, reg_param = 0, theta_init = None, intercept = True):
        n = features_mat.shape[0]
        p = features_mat.shape[1]
        k = rewards_mat.shape[1]
        
        if theta_init is None:
            theta = np.zeros((p, k))
        else:
            theta = theta_init
        for i in range(max_iter):
            scores_mat = features_mat@theta
            scores_mat = scores_mat
            exp_scores_mat = np.exp(scores_mat)
            exp_scores_mat_div_sum_exp = ((1/np.sum(exp_scores_mat, axis = 1))*exp_scores_mat.T).T
            temp_mat = (exp_scores_mat_div_sum_exp - exp_scores_mat_div_sum_exp ** 2)
            temp_mat = temp_mat * rewards_mat
            grad_theta_mat = np.average((temp_mat.reshape((n,1,k)) * features_mat.reshape((n,p,1))),axis = 0)
            theta = theta + (learning_rate)*(grad_theta_mat - reg_param * theta)
            theta = (theta.T - theta[:, 0]).T
        #print(exp_scores_mat_div_sum_exp)
        return theta

    def logistic_policy_stable(self, features_mat, rewards_mat, actions, pi_mat, max_iter = 1000, 
        learning_rate = 1, reg_param = 0, theta_init = None, intercept = True):
        """
        Train a logistic policy that maximize the total reward function
        reward: R = (sum_i (w_i*r_i/pi_i))/ sum_i (w_i/pi_i)
        where
        r_i = rewards_mat[i, A_i]
        w_i = exp(theta_{A_i} X_i) / sum_j(exp(theta_j X_i)) 
        w_i is an approximation of Indicator{theta_{A_i}X_i == max_j(theta_j X_i)}
        """
        n = features_mat.shape[0]
        p = features_mat.shape[1]
        k = rewards_mat.shape[1]

        if theta_init is None:
            theta = np.zeros((p, k))
        else:
            theta = theta_init

        for i in range(max_iter):
            # dim(scores_mat) = n * k
            scores_mat = features_mat@theta
            scores_mat = scores_mat - np.min(scores_mat)
            exp_scores_mat= np.exp(scores_mat)
            exp_scores_mat_normalize = ((1/np.sum(exp_scores_mat, axis = 1))*exp_scores_mat.T).T

            w_array = exp_scores_mat_normalize[range(n), actions.astype(int)]
            r_array = rewards_mat[range(n), actions.astype(int)]
            
            one_div_pi_array = 1/pi_mat[range(n), actions.astype(int)]
            r_div_pi_array = r_array*one_div_pi_array


            # dim(dR_dw) = sim(w) = n
            dR_dw = (r_div_pi_array * np.sum(w_array*one_div_pi_array) 
                - one_div_pi_array*np.sum(w_array*r_div_pi_array)) / (np.sum(w_array*one_div_pi_array) ** 2)

            # dw_dscores_mat[i,j] = dw_i/ dscores_mat[i,j]
            # dim(dw_dscores_mat) = (n, k)
            dw_dscores_mat = np.zeros((n, k))
            dw_dscores_mat[range(n), actions.astype(int)] += exp_scores_mat_normalize[range(n), actions.astype(int)]

            dw_dscores_mat -= exp_scores_mat_normalize[range(n), actions.astype(int)].reshape(n,1) \
            * exp_scores_mat_normalize

            temp_mat = dR_dw.reshape(-1,1) * dw_dscores_mat
            grad_theta_mat = np.average((temp_mat.reshape((n,1,k)) * features_mat.reshape((n,p,1))),axis = 0)
            theta = theta + (learning_rate)*(grad_theta_mat - reg_param * theta)
            theta = (theta.T - theta[:, 0]).T
        return theta

    def linear_regression_policy(self, features_mat, rewards_mat, actions):
        k = rewards_mat.shape[1]
        p = features_mat.shape[1]
        theta = np.zeros((p,k))
        for i in range(k):
            LR_model = LinearRegression(fit_intercept=False).fit(features_mat[actions == i,:],rewards_mat[actions == i, i])
            theta[:,i] = LR_model.coef_
        return theta

    def DR_logistic_policy(self, features_mat, rewards_mat, delta = 0.5, reg_param = 0,
                       logistic_learning_rate = 1, max_iter = 50, alpha_init = None, verbose = True):
        if alpha_init:
            alpha = alpha_init
        else:
            alpha = 4/(10*delta)
        n = features_mat.shape[0]
        tol = 1e-8
        last_reward = np.inf
        last_theta = None
        #theta_init = self.logistic_policy(features_mat, rewards_mat, reg_param = reg_param)
        theta_init = None

        for i in range(max_iter):
            rewardmat_DR = -np.exp(-rewards_mat/alpha)
            if last_theta is None:
                theta_logistic = self.logistic_policy(features_mat, rewardmat_DR, reg_param = reg_param, 
                    learning_rate = logistic_learning_rate, theta_init = theta_init)
            else:
                theta_logistic = self.logistic_policy(features_mat, rewardmat_DR, reg_param = reg_param, 
                    learning_rate = logistic_learning_rate, max_iter = 50, theta_init = last_theta)
            last_theta = theta_logistic
            theta_logistic_act = np.argmax(features_mat @ theta_logistic, axis = 1)
            Y_pi = rewards_mat[range(n),theta_logistic_act.astype(np.int)]
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
        return theta_logistic

    def DR_ind_logistic_policy(self, features_mat, rewards_mat, actions, pi_mat, reg_param = 0, delta = 0.5, 
                       alpha_init = None, max_iter = 50, verbose = True):
        if alpha_init:
            alpha = alpha_init
        else:
            alpha = 4/(10*delta)
        n = features_mat.shape[0]
        k = rewards_mat.shape[1]
        tol = 1e-8
        last_reward = np.inf
        last_theta = None
        act_div_pi = np.eye(k)[actions.astype(np.int)] / pi_mat

        IPW_mat = np.zeros_like(rewards_mat)
        IPW_mat[np.arange(n),actions.astype(np.int)] = (rewards_mat/pi_mat)[np.arange(n),actions.astype(np.int)]
        theta_init = self.logistic_policy(features_mat, IPW_mat, reg_param = reg_param)

        for i in range(max_iter):
            rewardmat_DR = -np.exp(-rewards_mat/alpha) * act_div_pi
            if last_theta is None:
                theta_logistic = self.logistic_policy(features_mat, rewardmat_DR, reg_param = reg_param, 
                    theta_init = theta_init)
            else:
                theta_logistic = self.logistic_policy(features_mat, rewardmat_DR, reg_param = reg_param, 
                    max_iter = 100, theta_init = last_theta)
            last_theta = theta_logistic
            theta_logistic_act = np.argmax(features_mat @ theta_logistic, axis = 1)
            Y_pi = rewards_mat[range(n),theta_logistic_act.astype(np.int)]
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
        return theta_logistic

    def DR_ind_logistic_policy_stable(self, features_mat, rewards_mat, actions, pi_mat, reg_param = 0, delta = 0.5, 
                       max_iter = 50, logistic_learning_rate = 1, alpha_init = None, verbose = True):
        if alpha_init:
            alpha = alpha_init
        else:
            alpha = 4/(10*delta)
        n = features_mat.shape[0]
        k = rewards_mat.shape[1]
        tol = 1e-8
        last_reward = -np.inf
        last_theta = None

        IPW_mat = np.zeros_like(rewards_mat)
        IPW_mat[np.arange(n),actions.astype(np.int)] = (rewards_mat/pi_mat)[np.arange(n),actions.astype(np.int)]
        theta_init = self.logistic_policy(features_mat, IPW_mat, reg_param = reg_param)

        for i in range(max_iter):
            rewardmat_DR = -np.exp(-rewards_mat/alpha)
            if last_theta is None:
                theta_logistic = self.logistic_policy_stable(features_mat, rewardmat_DR, actions, pi_mat, 
                    learning_rate = logistic_learning_rate, reg_param = reg_param, theta_init = theta_init)
            else:
                theta_logistic = self.logistic_policy_stable(features_mat, rewardmat_DR, actions, pi_mat,
                    learning_rate = logistic_learning_rate, reg_param = reg_param, max_iter = 200, theta_init = last_theta)
            last_theta = theta_logistic
            theta_logistic_act = np.argmax(features_mat @ theta_logistic, axis = 1)
            Y_pi = rewards_mat[range(n),theta_logistic_act.astype(np.int)]
            W_n = np.mean(np.exp(-Y_pi/alpha))
            d_W_n = np.mean(Y_pi *np.exp(-Y_pi/alpha)/ (alpha*alpha))
            d2_W_n = np.mean(np.exp(-Y_pi/alpha) * ((Y_pi**2)/(alpha**4)-(2*Y_pi)/(alpha**3)))
            DR_reward = -alpha*np.log(W_n)-alpha*delta
            d_DR_reward = -np.log(W_n) - delta - alpha*d_W_n/W_n
            d2_DR_reward = -2*d_W_n/W_n - alpha * (d2_W_n*W_n - d_W_n**2)/(W_n**2)
            alpha = alpha - d_DR_reward/d2_DR_reward
            if verbose:
                print("iter={:d},  alpha={:.4f}, DR_reward = {:.4f}".format(i,alpha,DR_reward))
            if (abs(last_reward - DR_reward) < tol or last_reward > DR_reward):
                if verbose:
                    print('Early break...')
                break
            last_reward = DR_reward
        return theta_logistic

