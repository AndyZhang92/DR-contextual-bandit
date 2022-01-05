class Configuration(object):
    # EXPERIMENT CONFIGURATION
    def __init__(self, mode = "ellipse"):

        if mode == "ellipse":
            self.mode = "ellipse"
            self.n_best = 50000 # number of observations to calculate best tree
            self.n_test = 20000 # number of observations to test learned trees
            self.p = 10 # number of features
            
            self.f1 = 5 # first essential feature
            self.f2 = 7 # second essential feature
            
            self.train_shape = {"r11": 0.6, "r12": 0.35, "r21": 0.4, "r22": 0.35}
            self.test_shape = {"r11": 0.7, "r12": 0.25, "r21": 0.5, "r22": 0.25}
            
            self.k = 3 # number of actions
            self.noise_level = 2
            self.delta = 0.1 #size of uncertainty ball
            
            self.num_exp = 500
            self.n_exp_list = [1000,1500,2000,3000,4000,5000,7000,10000,20000]

        if mode == "linear":
            self.mode = "linear"
            self.p = 10 # number of features
            self.f1 = 5 # first essential feature
            self.f2 = 7 # second essential feature

            self.k = 3 # number of actions
            self.noise_level = 2
            self.delta = 0.1 #size of uncertainty ball