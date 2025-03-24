class Config():

  def __init__(self, n_pairs, n_users, feature_dim, 
               noise_observations=False, random_seed=88, svd_init=True,
               svd_cheat=False, batch_size=8, gradient_accumulation_steps=2,
               max_iterations = 50000, svd_fit_iterations = 50,
               svd_fit_threshold = 1e-4, lr = 1e-6, train_frac = 0.9,
               stopping_thresh = 1e-6, stopping_patience=10, inv_temperature = 1.0, name=None,
               tuning_frac = 0.5, train_user_frac=0.5,
               model_feature_dim=None, use_svd_rank=False, regularization=None, regularization_strength=0.0, save_model=True, noise=0.0, model_name=None):
    self.n_pairs = n_pairs
    self.n_users = n_users
    self.feature_dim = feature_dim
    self.random_seed = random_seed
    self.noise_observations = noise_observations
    self.batch_size = batch_size
    self.svd_init = svd_init
    self.max_iterations = max_iterations
    self.svd_fit_iterations = svd_fit_iterations
    self.svd_fit_threshold = svd_fit_threshold
    self.lr = lr
    self.train_frac = train_frac
    self.stopping_thresh = stopping_thresh
    self.name = name
    self.stopping_patience = stopping_patience
    self.inv_temperature = inv_temperature
    self.tuning_frac = tuning_frac
    self.train_user_frac = train_user_frac
    self.model_feature_dim=model_feature_dim
    self.use_svd_rank=use_svd_rank
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.regularization = regularization
    self.regularization_strength = regularization_strength
    self.save_model = save_model
    self.noise = noise
    self.model_name = model_name
  def __str__(self):
    if self.name is None:
      return f'model_{self.model_name}-npairs_{self.n_pairs}-nusers_{self.n_users}-fdim_{self.feature_dim}-svd_{self.svd_init}-batch_{self.batch_size*self.gradient_accumulation_steps}-seed_{self.random_seed}-maxiter_{self.max_iterations}'
    else:
      return self.name
    
class PALConfig():

  def __init__(self, n_pairs, n_users, feature_dim, 
                noise_observations=False, random_seed=88, batch_size=8, gradient_accumulation_steps=2,
                max_iterations = 50000, svd_fit_iterations = 50,
                svd_fit_threshold = 1e-4, lr = 1e-6, train_frac = 0.9,
                stopping_thresh = 1e-6, stopping_patience=10, inv_temperature = 1.0, name=None,
                tuning_frac = 0.5, train_user_frac=0.5,
                model_feature_dim=None, save_model=True, noise=0.0, model_name=None, num_cal_coords=None):
    self.n_pairs = n_pairs
    self.n_users = n_users
    self.feature_dim = feature_dim
    self.random_seed = random_seed
    self.noise_observations = noise_observations
    self.batch_size = batch_size
    self.max_iterations = max_iterations
    self.svd_fit_iterations = svd_fit_iterations
    self.svd_fit_threshold = svd_fit_threshold
    self.lr = lr
    self.train_frac = train_frac
    self.stopping_thresh = stopping_thresh
    self.name = name
    self.stopping_patience = stopping_patience
    self.inv_temperature = inv_temperature
    self.tuning_frac = tuning_frac
    self.train_user_frac = train_user_frac
    self.model_feature_dim=model_feature_dim
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.save_model = save_model
    self.noise = noise
    self.model_name = model_name
    self.num_cal_coords = num_cal_coords
  def __str__(self):
    if self.name is None:
      return f'PAL_{self.model_name}-npairs_{self.n_pairs}-nusers_{self.n_users}-fdim_{self.feature_dim}-batch_{self.batch_size*self.gradient_accumulation_steps}-seed_{self.random_seed}-maxiter_{self.max_iterations}-num_cal_coords_{self.num_cal_coords}'
    else:
      return self.name
  

class VLPConfig():

  def __init__(self, n_pairs, n_users, 
               noise_observations=False, random_seed=88, batch_size=8, gradient_accumulation_steps=2,
               max_iterations = 50000, lr = 1e-6, train_frac = 0.9,
               stopping_thresh = 1e-6, stopping_patience=10, inv_temperature = 1.0, name=None,
               tuning_frac = 0.5, train_user_frac=0.5,
               model_feature_dim=None, use_svd_rank=False,save_model=True, noise=0.0, model_name=None):
    self.n_pairs = n_pairs
    self.n_users = n_users
    self.random_seed = random_seed
    self.noise_observations = noise_observations
    self.batch_size = batch_size
    self.max_iterations = max_iterations
    self.lr = lr
    self.train_frac = train_frac
    self.stopping_thresh = stopping_thresh
    self.name = name
    self.stopping_patience = stopping_patience
    self.inv_temperature = inv_temperature
    self.tuning_frac = tuning_frac
    self.train_user_frac = train_user_frac
    self.model_feature_dim=model_feature_dim
    self.use_svd_rank=use_svd_rank
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.save_model = save_model
    self.noise = noise
    self.model_name = model_name
  def __str__(self):
    if self.name is None:
      return f'VLP_{self.model_name}-batch_{self.batch_size*self.gradient_accumulation_steps}-seed_{self.random_seed}-maxiter_{self.max_iterations}'
    else:
      return self.name
    


class AvgUserConfig():

  def __init__(self, n_pairs, n_users, 
               noise_observations=False, 
               random_seed=88, 
               batch_size=8, 
               gradient_accumulation_steps=2,
               max_iterations = 50000, 
               lr = 1e-6, 
               train_frac = 0.9,
               stopping_thresh = 1e-6, 
               stopping_patience=10, 
               inv_temperature = 1.0,
               name=None,
               tuning_frac = 0.5, 
               train_user_frac=0.5,
               save_model=True, noise=0.0):
    self.n_pairs = n_pairs
    self.n_users = n_users
    self.random_seed = random_seed
    self.noise_observations = noise_observations
    self.batch_size = batch_size
    self.max_iterations = max_iterations
    self.lr = lr
    self.train_frac = train_frac
    self.stopping_thresh = stopping_thresh
    self.name = name
    self.stopping_patience = stopping_patience
    self.inv_temperature = inv_temperature
    self.tuning_frac = tuning_frac
    self.train_user_frac = train_user_frac
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.save_model = save_model
    self.noise = noise
  def __str__(self):
    if self.name is None:
      return f'avg_user_npairs_{self.n_pairs}-nusers_{self.n_users}-batch_{self.batch_size*self.gradient_accumulation_steps}-seed_{self.random_seed}-maxiter_{self.max_iterations}-PRISM'
    else:
      return self.name

class IndividualUserConfig():
    def __init__(self, 
                n_pairs,
                n_users,
                noise_observations=False,
                random_seed=88,
                batch_size=2,
                gradient_accumulation_steps=32,
                max_iterations=50,
                lr=0.001,
                train_frac=0.9,
                stopping_thresh=0.005,
                stopping_patience=5,
                name=None,
                noise=0.0,
                max_train_points_per_user=None,
                dataset=None):
        self.n_pairs = n_pairs
        self.n_users = n_users
        self.random_seed = random_seed
        self.noise_observations = noise_observations
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.lr = lr
        self.train_frac = train_frac
        self.stopping_thresh = stopping_thresh
        self.stopping_patience = stopping_patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.noise = noise
        self.name = name
        self.max_train_points_per_user = max_train_points_per_user
        self.dataset = dataset
    def __str__(self):
        if self.name is None:
            base_name = f'individual_user_frozen_npairs_{self.n_pairs}-nusers_{self.n_users}-batch_{self.batch_size*self.gradient_accumulation_steps}-seed_{self.random_seed}-maxiter_{self.max_iterations}-'
            if self.max_train_points_per_user:
                base_name += f'-max_points_{self.max_train_points_per_user}'
            if self.dataset:
                base_name += f'-{self.dataset}'
            return base_name
        else:
            return self.name