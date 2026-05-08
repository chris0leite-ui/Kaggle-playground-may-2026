# U3 probe — train/test split structure (2026-05-04)

Total (Race, Driver) groups present in BOTH train AND test: 13185

## Headline
- mean alt-ratio per group: **0.4471**
- median: 0.4227
- groups with alt-ratio > 0.40 (i.i.d.-like): 7686 / 13185
- groups with alt-ratio < 0.05 (contiguous-like): 0 / 13185
- groups where train lap-range OVERLAPS test lap-range: 12212 / 13185

## Verdict
**i.i.d. row-level split** — within-group laps interleave between train and test. lead(PitStop) features are computable on test directly. GroupKFold(Race) is the right anchor.

## Sample sequences (5 random groups, T=train row, t=test row, sorted by lap)

- (Azerbaijan Grand Prix   , D116 ): n= 27, alt= 11, ratio=0.423, seq=TTtTTttTttTTTTTttTTTTtTTTTt
- (Emilia Romagna Grand Pri, D127 ): n= 44, alt= 19, ratio=0.442, seq=TTTTtTtTTTTtTtTTTtTTTtTTTTTttTTTtTTtTTTTTTtt
- (Monaco Grand Prix       , D198 ): n= 39, alt= 20, ratio=0.526, seq=tTTtTtTTTTtTTTTttTTtTTTTTttTTttTtTTTtTt
- (Australian Grand Prix   , MAS  ): n=107, alt= 52, ratio=0.491, seq=TtTttTTTTTtTTTTttTTTTttTTtTTtTTTTtTTTTtTTtTtTTtTTTTTTTTTtTTTTTTtttTtTTTTtTtTTTtT
- (Miami Grand Prix        , D307 ): n= 25, alt= 10, ratio=0.417, seq=tTTtTTTtTTTTTTTtTTTTTTtTt
