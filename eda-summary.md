# EDA summary — playground-series-s6e5
- train rows: 439,140  test rows: 188,165
- numeric features: 11  categorical: 3
- missingness in train: 0.0000
- class priors: {0.0: np.float64(0.8010201757981509), 1.0: np.float64(0.19897982420184906)}
- top-5 numeric signals (F): {'TyreLife': 17546.011850958694, 'LapNumber': 16970.715311627944, 'Stint': 9208.832202064554, 'RaceProgress': 7902.413956931822, 'Cumulative_Degradation': 6175.223863996355}
- top-5 categorical signals (chi²): {'Compound': 15168.287862825098, 'Race': 7752.511172838999, 'Driver': 4291.609291236513}
- top-3 train/test drift (z): [{'col': 'Cumulative_Degradation', 'z_diff': 0.005639950635808426}, {'col': 'Position', 'z_diff': 0.005238829797387406}, {'col': 'LapTime_Delta', 'z_diff': 0.004742473995462314}]
