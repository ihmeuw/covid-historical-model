import pandas as pd

import db_queries

from covid_historical_model.utils.misc import parent_inheritance


def obesity(hierarchy: pd.DataFrame) -> pd.DataFrame:
    data = db_queries.get_covariate_estimates(
        gbd_round_id=6,
        decomp_step='iterative',
        covariate_id=455,
        year_id=2019,
    )
    
    # just averaging sexes here...
    data = (data
            .groupby('location_id')['mean_value'].mean()
            .rename('obesity'))
    
    # pass down hierarchy
    data = data.to_frame()
    data = parent_inheritance(data, hierarchy)
    data = data.squeeze()
    
    return data


def age_metadata() -> pd.DataFrame:
    data = db_queries.get_age_metadata(
        gbd_round_id=6,
        age_group_set_id=12,
    )
    
    data = data.loc[data['age_group_years_start'] >= 5, ['age_group_years_start', 'age_group_years_end', 'age_group_id']]
    data.loc[data['age_group_years_end'] != 125, 'age_group_years_end'] -= 1
    
    data = pd.concat([pd.DataFrame({'age_group_years_start': 0, 'age_group_years_end': 4, 'age_group_id': 1}, index=[0]),
                      data])
    
    for col in data.columns:
        data[col] = data[col].astype(int)
    
    return data
