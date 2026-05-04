# EDA summary — playground-series-s6e5
- train rows: 439,140  test rows: 188,165
- numeric features: 11  categorical: 3
- missingness in train: 0.0000
- target stats: {'count': 219570.0, 'mean': 0.19866557362116866, 'std': 0.39899660214675975, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 0.0, 'max': 1.0}
- top-5 numeric signals (F): {'TyreLife': 17828.249604053784, 'LapNumber': 16847.883179304765, 'Stint': 9024.062516064872, 'RaceProgress': 7829.042346453793, 'Cumulative_Degradation': 6558.493660963927}
- top-3 train/test drift (z): [{'col': 'LapTime_Delta', 'z_diff': 0.0060348319422143665}, {'col': 'Year', 'z_diff': 0.005053079827887621}, {'col': 'RaceProgress', 'z_diff': 0.004180826566018013}]
