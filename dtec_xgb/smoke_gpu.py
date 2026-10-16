"""Tiny synthetic GPU fit/predict check; no clinical data, Optuna or Slurm submission."""
from datetime import datetime
import os

import numpy as np
from xgboost import XGBRegressor

from data import ROOT, write_json
from worker import verify_booster_device


def main():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(44, 20))
    y = np.repeat([3., 8., 13., 18.], 11)
    result = []
    for columns in [list(range(20)), list(range(5))]:
        model = XGBRegressor(objective='reg:squarederror', learning_rate=.03, max_depth=4,
                             n_estimators=3, tree_method='hist', device='cuda:0', random_state=42, n_jobs=2)
        model.fit(X[11:][:, columns], y[11:])
        device = verify_booster_device(model, 'cuda:0')
        predictions = model.predict(X[:11][:, columns])
        assert predictions.shape == (11,) and np.isfinite(predictions).all()
        result.append(dict(features=len(columns), device=device, predictions=len(predictions)))
    directory = ROOT / 'validation'
    directory.mkdir(exist_ok=True)
    path = directory / f'gpu_smoke_{datetime.now():%Y%m%dT%H%M%S}.json'
    write_json(path, dict(synthetic=True, real_data_training=False,
                         visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), results=result))
    print('SYNTHETIC GPU SMOKE PASSED', path)


if __name__ == '__main__':
    main()
