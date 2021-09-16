import numpy as np

COVARIATE_PRIORS = {
    # 'obesity': {'prior_beta_uniform': np.array([np.log(1.3), np.log(1.3)])},
    'obesity':
        # 1.44 (1.16 - 1.79)
        {'prior_beta_gaussian': np.array([np.log(1.44),
                                          ((np.log(1.79) - np.log(1.16)) / 3.92) * 5]),
         'prior_beta_uniform': np.array([0, np.inf]),},
    'smoking':
        # 1.11 (0.93 - 1.33)
        {'prior_beta_gaussian': np.array([np.log(1.11),
                                          ((np.log(1.33) - np.log(0.93)) / 3.92) * 5]),
         'prior_beta_uniform': np.array([0, np.inf]),},
    'diabetes':
        # 1.1 (1 - 1.21)
        {'prior_beta_gaussian': np.array([np.log(1.1),
                                          ((np.log(1.21) - np.log(1.)) / 3.92) * 5]),
         'prior_beta_uniform': np.array([0, np.inf]),},
    'cancer':
        # 1.25 (1.1 - 1.41)
        {'prior_beta_gaussian': np.array([np.log(1.25),
                                          ((np.log(1.41) - np.log(1.1)) / 3.92) * 5]),
         'prior_beta_uniform': np.array([0, np.inf]),},
    'copd':
        # 1.07 (0.96 - 1.2)
        {'prior_beta_gaussian': np.array([np.log(1.07),
                                          ((np.log(1.2) - np.log(0.96)) / 3.92) * 5]),
         'prior_beta_uniform': np.array([0, np.inf]),},
    'cvd':
        # stroke: 1.2 (1.05 - 1.37)
        # heart failure: 1.22 (1.07 - 1.38)
        {'prior_beta_gaussian': np.array([np.log(1.21),
                                          ((np.log(1.38) - np.log(1.05)) / 3.92) * 5]),
         'prior_beta_uniform': np.array([0, np.inf]),},
    'ckd':
        # 0.88 (0.77 - 1.01) --> moving to 1
        {'prior_beta_gaussian': np.array([np.log(1.),
                                          ((np.log(1.01) - np.log(0.77)) / 3.92) * 5]),
         'prior_beta_uniform': np.array([0, np.inf]),},
    'uhc':
        {'prior_beta_uniform': np.array([-np.inf, 0])},
    'haq':
        {'prior_beta_uniform': np.array([-np.inf, 0])},
}
