"""Small protocol checks; no clinical training or GPU allocation."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import runner_core as core


class FakeBooster:
    def save_config(self):
        return json.dumps({'learner': {'generic_param': {'device': 'cuda:0'}}})


class FakeModel:
    def __init__(self, **params):
        self.feature_importances_ = np.array([0.25, 0.75])

    def fit(self, X, y):
        self.training_ids = set(X[:, 0])

    def predict(self, X):
        assert not self.training_ids.intersection(X[:, 0])
        return np.arange(len(X), dtype=np.float32)

    def get_booster(self):
        return FakeBooster()


class ProtocolTests(unittest.TestCase):
    def test_train_only_shap_and_all_patient_predictions(self):
        X = np.array([[pid, s] for pid in range(4) for s in range(3)], dtype=float)
        groups = np.repeat(np.array(['0','1','2','3']), 3)
        y = groups.astype(float)
        seen = []
        def importance(model, training):
            self.assertEqual(set(training[:, 0]), model.training_ids)
            self.assertEqual(len(training), 9)
            seen.append(model.training_ids)
            return np.array([0.8, 0.2])
        with tempfile.TemporaryDirectory() as d, patch.object(core, 'XGBRegressor', FakeModel), \
                patch.object(core, 'mean_abs_training_shap', importance), patch.object(core, 'CURRENT_TOP_K', 5):
            results, imp, _ = core.run_lopo(X, y, groups, np.array(['0','1','2','3']), {str(i):i for i in range(4)},
                                            Path(d), 'full_feature', {})
            self.assertEqual(len(results), 4)
            self.assertEqual(len(seen), 4)
            np.testing.assert_array_equal(results.predicted_score, np.ones(4))
            np.testing.assert_allclose(imp.feat_0, 0.8)
            self.assertEqual(len(list(Path(d).glob('*.pkl'))),0)
            with patch.object(core, 'mean_abs_training_shap', side_effect=AssertionError('Top-K SHAP forbidden')):
                selected, _, best = core.run_lopo(X, y, groups, np.array(['0','1','2','3']), {str(i):i for i in range(4)}, Path(d), 'top5', {}, save_best=True)
                self.assertEqual(len(selected),4)
                self.assertEqual(best['rmse'],selected.rmse.min())
                self.assertEqual(len(list(Path(d).glob('*.pkl'))),1)

    def test_all_final_lopo_once_without_model_persistence(self):
        X=np.array([[pid,s] for pid in range(4) for s in range(3)],dtype=float)
        ids=np.array(['0','1','2','3']);groups=np.repeat(ids,3)
        data=(X,groups.astype(float),groups,ids,{str(i):i for i in range(4)})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input.csv';source.write_text('fixture')
            with patch.object(core,'ROOT',root),patch.object(core,'CONFIGS',{'mfcc':(source,'features')}), \
                    patch.object(core,'tune',return_value={'learning_rate':0.01,'max_depth':4,'n_estimators':100}), \
                    patch.object(core,'XGBRegressor',FakeModel), \
                    patch.object(core,'mean_abs_training_shap',return_value=np.array([0.8,0.2])), \
                    patch.object(core,'run_lopo',wraps=core.run_lopo) as lopo:
                row=core.experiment('mfcc',None,data)
            self.assertEqual(lopo.call_count,1)
            self.assertEqual(lopo.call_args.args[6],'all_feature')
            self.assertEqual(row['num_patients'],4)
            self.assertEqual(row['num_features_used'],2)
            self.assertEqual(len(list(root.rglob('*.pkl'))),0)

    def test_reference_parameters_and_parsing(self):
        core.verify_reference()
        self.assertEqual(core.TRIALS, 20)
        self.assertEqual(core.GPU_DEVICES, ('cuda:0',))
        self.assertEqual(core.MAX_WORKERS, 1)

    def test_unchanged_shap_helper(self):
        root=Path(__file__).resolve().parent
        old=root.parents[1]/'shap_works/optuna_xgb_configurable_shap.py'
        if not old.exists():
            old=root/'reference/shap_helper.py'
        def function(path):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='mean_abs_training_shap')
        self.assertEqual(ast.dump(function(old)),ast.dump(function(root/'shap_ranking.py')))

    def test_real_cpu_shap_smoke(self):
        from xgboost import XGBRegressor
        from shap_ranking import mean_abs_training_shap
        X=np.random.default_rng(42).normal(size=(20, 3))
        model=XGBRegressor(n_estimators=3,max_depth=2,device='cpu',n_jobs=1).fit(X,X[:,0])
        values=mean_abs_training_shap(model,X)
        self.assertEqual(values.shape,(3,))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values>=0).all())


if __name__=='__main__':
    unittest.main()
