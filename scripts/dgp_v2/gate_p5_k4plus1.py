"""K=4+1 gate for the Phase-5 pure-orig-stint candidate."""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import probe

K4_BASES_OOF = [
    "scripts/artifacts/oof_d17_h1d_yekenot_full_strat.npy",
    "scripts/artifacts/oof_p1_single_cb_v4_gpu_strat.npy",
    "scripts/artifacts/oof_f1_hgbc_deep_strat.npy",
    "scripts/artifacts/oof_d16_orig_continuous_only_strat.npy",
]
K4_BASES_TEST = [
    "scripts/artifacts/test_d17_h1d_yekenot_full_strat.npy",
    "scripts/artifacts/test_p1_single_cb_v4_gpu_strat.npy",
    "scripts/artifacts/test_f1_hgbc_deep_strat.npy",
    "scripts/artifacts/test_d16_orig_continuous_only_strat.npy",
]
PRIMARY_OOF = "scripts/artifacts/oof_K4_fwd_pathb.npy"
PRIMARY_TEST = "scripts/artifacts/test_K4_fwd_pathb.npy"

CAND = "p5_pure_orig_stint"

probe.gate(
    name=CAND,
    oof_path=f"scripts/artifacts/oof_{CAND}_strat.npy",
    test_path=f"scripts/artifacts/test_{CAND}_strat.npy",
    primary_oof_path=PRIMARY_OOF,
    primary_test_path=PRIMARY_TEST,
    train_csv="data/train.csv",
    min_meta=True,
    min_meta_pool_oofs=K4_BASES_OOF,
    min_meta_pool_tests=K4_BASES_TEST,
)
