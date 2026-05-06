
## TL;DR

- Direct FM (k=8, 6 fields) val AUC 0.9032; embeddings extractable
- Compound cosine matrix shows whether FM groups SOFT/MED/HARD as a tyre-spectrum (cosine should be high among the 3 dry compounds, low against WET/INTER)
- Driver PCA: visible 'aggressive' clusters at high pit-rate end (top-1/3)
- Field-pair strength matrix: highest pair magnitude reveals which FM interactions drive the +3bp LB lift; weakest pairs are candidates to drop in the next FM iteration
- FM minus GBDT mean by (Compound, Stint) shows where they disagree — a base specialized on those segments could lift orthogonality further
# Phase E — FM embedding visualization

- Direct (un-hashed) FM, k=8, fold-0 val AUC=0.90324
- field sizes: {'Driver': 887, 'Compound': 5, 'Race': 26, 'Year': 4, 'Stint': 8, 'Compound_prev': 6}

## Compound cosine matrix (does FM learn SOFT-MED-HARD ordering?)

```
              HARD  INTERMEDIATE  MEDIUM  SOFT   WET
HARD          1.00          0.39   -0.67  0.20  0.02
INTERMEDIATE  0.39          1.00   -0.44 -0.12  0.59
MEDIUM       -0.67         -0.44    1.00 -0.36 -0.10
SOFT          0.20         -0.12   -0.36  1.00 -0.37
WET           0.02          0.59   -0.10 -0.37  1.00
```


## Field-pair interaction magnitude

```
               Driver  Compound   Race   Year  Stint  Compound_prev
Driver          0.000     0.128  0.204  0.361  0.276          0.121
Compound        0.128     0.000  0.144  0.198  0.252          0.124
Race            0.204     0.144  0.000  0.343  0.317          0.129
Year            0.361     0.198  0.343  0.000  0.386          0.159
Stint           0.276     0.252  0.317  0.386  0.000          0.183
Compound_prev   0.121     0.124  0.129  0.159  0.183          0.000
```


## FM vs GBDT mean-prediction by Compound × Stint (fold-0)

```
                        n    tgt     fm     gb  delta
Compound     Stint                                   
HARD         1       3584  0.078  0.078  0.076  0.002
             2      19892  0.391  0.399  0.390  0.009
             3       9344  0.314  0.318  0.315  0.003
             4       1205  0.232  0.229  0.231 -0.002
             5        156  0.135  0.128  0.118  0.009
INTERMEDIATE 1       2050  0.100  0.132  0.105  0.028
             2        445  0.321  0.343  0.316  0.027
             3        402  0.323  0.340  0.325  0.014
             4        394  0.038  0.074  0.039  0.035
MEDIUM       1      33603  0.035  0.032  0.035 -0.003
             2       5160  0.439  0.459  0.452  0.007
             3       2407  0.271  0.276  0.268  0.009
             4        910  0.147  0.146  0.150 -0.004
             5        170  0.059  0.079  0.052  0.028
SOFT         1       3707  0.242  0.262  0.239  0.023
             2        647  0.178  0.186  0.188 -0.002
             3       1565  0.137  0.159  0.143  0.016
             4       1216  0.165  0.207  0.169  0.038
             5        487  0.021  0.035  0.017  0.018
WET          1        271  0.018  0.019  0.023 -0.004
```

