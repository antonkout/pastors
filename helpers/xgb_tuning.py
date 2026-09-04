"""
XGBoost tuning + final fit utilities.

Two public functions:
    tune_xgb(X, y, n_trials=200) -> (best_params, best_iter, best_cv_r2)
    fit_final(X, y, params, n_estimators) -> fitted XGBRegressor

Design choices:
  - CV is group-aware when `groups` is passed (leave-site-out / spatial
    GroupKFold), else random KFold (within-image, leaky) as before.
  - Median best_iter * 1.1 for the final n_estimators.
  - Wide Optuna search space.
"""

import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import r2_score


def _cv_splits(X, y, groups, n_splits, seed=42):
    """Yield (train_idx, val_idx). Group-aware if `groups` is given."""
    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        splitter = (LeaveOneGroupOut() if n_splits >= n_groups
                    else GroupKFold(n_splits=n_splits))
        return list(splitter.split(X, y, groups))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X))


def _is_huber(objective):
    return objective == 'reg:pseudohubererror'


def _objective(trial, X, y, groups=None, n_splits=5,
               objective='reg:squarederror'):
    params = dict(
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        gamma=trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
        n_estimators=3000, early_stopping_rounds=50,
        objective=objective, random_state=42, n_jobs=-1,
        eval_metric='mphe' if _is_huber(objective) else 'rmse',
    )
    # huber_slope (delta) sets where the loss switches L2 -> L1; tune it so the
    # robustness cutoff is fitted to the balance's tail rather than guessed.
    if _is_huber(objective):
        params['huber_slope'] = trial.suggest_float('huber_slope', 0.1, 5.0, log=True)
    scores, iters = [], []
    for tr, va in _cv_splits(X, y, groups, n_splits):
        m = XGBRegressor(**params)
        m.fit(X[tr], y[tr],
              eval_set=[(X[va], y[va])], verbose=False)
        scores.append(r2_score(y[va], m.predict(X[va])))
        iters.append(m.get_booster().best_iteration)
    trial.set_user_attr('best_iter', int(np.median(iters)))
    return float(np.mean(scores))


def _log_trial(study, trial):
    if trial.number % 20 == 0:
        print(f"  Trial {trial.number}: R² = {trial.value:.3f} "
              f"(best: {study.best_value:.3f})")


def tune_xgb(X, y, groups=None, n_trials=200, n_splits=5, seed=42,
             objective='reg:squarederror', verbose=True):
    """Run Optuna search. Returns (best_params, best_iter, best_cv_r2).

    If `groups` is given, CV holds whole groups out (leave-site-out when
    n_splits >= n_groups, else spatial GroupKFold); otherwise random KFold.

    `objective` selects the training loss. Use 'reg:pseudohubererror' for a
    robust fit that resists extreme balances (huber_slope is auto-tuned);
    best_params then carries huber_slope, but NOT the objective itself, so
    pass the same `objective` to fit_final.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    callbacks = [_log_trial] if verbose else []
    study.optimize(
        lambda t: _objective(t, X, y, groups=groups, n_splits=n_splits,
                             objective=objective),
        n_trials=n_trials, callbacks=callbacks,
    )
    best_iter = int(study.best_trial.user_attrs['best_iter'] * 1.1)
    return study.best_params, best_iter, study.best_value


def fit_final(X, y, params, n_estimators, seed=42,
              objective='reg:squarederror'):
    """Refit on all available data with the tuned params.

    Pass the same `objective` used in tune_xgb; `params` may contain
    huber_slope (harmless for the squared-error objective).
    """
    model = XGBRegressor(
        **params, n_estimators=n_estimators, objective=objective,
        random_state=seed, n_jobs=-1,
    )
    model.fit(X, y)
    return model