import numpy as np
from src.utils.seq_preproc import fit_train_only_stats, transform_with_stats, make_rf_features


def test_preproc_and_feature_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4, 3)).astype(np.float32)
    X[0, 0, 1] = np.nan  # inject NaN
    stats = fit_train_only_stats(X)
    Xs = transform_with_stats(X, stats)

    assert Xs.shape == X.shape
    F_flat = make_rf_features(Xs, "flatten")
    assert F_flat.shape == (10, 4 * 3)

    F_agg = make_rf_features(Xs, "agg")
    assert F_agg.shape == (10, 3 * 4)
