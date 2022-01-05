import numpy as np
import scipy.optimize

class linearPOEMPolicyLearner(object):
    def __init__(self):
        pass
    
    def POEM_objective(self, theta, features_mat, rewards_mat, actions, pi_mat, max_clip, var_reg):
        n = features_mat.shape[0]
        p = features_mat.shape[1]
        k = rewards_mat.shape[1]
        theta_mat = theta.reshape((p,k))
        scores_mat = features_mat@theta_mat # dim(scores_mat) = n * k
        scores_mat = scores_mat - np.min(scores_mat)
        scores_mat = np.clip(scores_mat, a_min = None, a_max = 100)
        exp_scores_mat = np.exp(scores_mat)
        h_array = exp_scores_mat[range(n), actions.astype(int)] / np.max(exp_scores_mat, axis = 1)
        cliped_u = (rewards_mat[range(n), actions.astype(int)] * 
                    np.clip(h_array/pi_mat[range(n), actions.astype(int)], a_min = None, a_max = 10))
        CRM_obj = np.mean(cliped_u) + var_reg * np.sqrt(np.var(cliped_u, ddof = 1)/n)
        
        return -CRM_obj
    
    def POEM_policy(self, features_mat, rewards_mat, actions, pi_mat, max_clip, var_reg, 
                    tol = 1e-7, max_iter = 100, theta_init = None, verbose = True):
        n = features_mat.shape[0]
        p = features_mat.shape[1]
        k = rewards_mat.shape[1]
        ops = {'maxiter': max_iter, 'disp': verbose, 'gtol': tol, 'ftol': tol, 'maxcor': 50}
        if theta_init is None:
            theta_init = np.zeros((p,k))
            
        Result = scipy.optimize.minimize(fun = self.POEM_objective, 
                                         args = (features_mat, rewards_mat, actions, pi_mat, max_clip, var_reg), 
                                         x0 = theta_init, method = 'L-BFGS-B', jac = '2-point', tol = tol, options = ops)
        res_theta = Result.x.reshape((p,k))
        return res_theta
        
    