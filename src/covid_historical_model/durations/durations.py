import numpy as np


def make_duration_dict(exposure_to_admission: int,
                       exposure_to_seroconversion: int,
                       admission_to_death: int,):
    durations = {
        # exposure
        'exposure_to_case': exposure_to_admission,
        'exposure_to_admission': exposure_to_admission,
        'exposure_to_seroconversion': exposure_to_seroconversion,
        'exposure_to_death': exposure_to_admission + admission_to_death,
    }
    
    # CASE_TO_DEATH = EXPOSURE_TO_DEATH - EXPOSURE_TO_CASE
    # PCR_TO_SERO = EXPOSURE_TO_SEROPOSITIVE - EXPOSURE_TO_CASE

    # ADMISSION_TO_SERO = EXPOSURE_TO_SEROPOSITIVE - EXPOSURE_TO_ADMISSION
    # ADMISSION_TO_DEATH = EXPOSURE_TO_DEATH - EXPOSURE_TO_ADMISSION

    # SERO_TO_DEATH = EXPOSURE_TO_DEATH - EXPOSURE_TO_SEROPOSITIVE
    
    return durations


def get_duration_dist(n_samples: int):
    durations = [make_duration_dict(eta, ets, atd)
     for n, (eta, ets, atd) in enumerate(zip(
         np.random.choice(list(range(10, 13)), size=n_samples).tolist(),  # exposure to case/admission
         np.random.choice(list(range(14, 18)), size=n_samples).tolist(),  # exposure to seroconversion
         np.random.choice(list(range(12, 15)), size=n_samples).tolist(),  # admission to death
     ))]
    
    return durations
