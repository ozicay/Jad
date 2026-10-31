"""Synthetic checks only; no patient metadata or downloaded weights required."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train_lopo as train
from pooling import pool_mean_variance

class ProtocolTests(unittest.TestCase):
    def test_pooling_matches_numpy_population_variance(self):
        hidden=torch.arange(3*1024,dtype=torch.float32).reshape(1,3,1024)/100
        actual=pool_mean_variance(hidden).numpy();reference=hidden[0].numpy().astype(np.float64)
        np.testing.assert_allclose(actual,np.r_[reference.mean(0),reference.var(0,ddof=0)],rtol=1e-6,atol=1e-6)
        self.assertFalse(np.allclose(actual[1024:],reference.var(0,ddof=1)))
        np.testing.assert_array_equal(pool_mean_variance(torch.ones(1,1,1024)).numpy()[1024:],np.zeros(1024))
        with self.assertRaises(ValueError):pool_mean_variance(torch.ones(2,3,1024))

    def test_loader_target_and_leading_zero_patient_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'features.csv'
            frame=pd.DataFrame({'patient_id':['001','001','002','002','003','003'],
                'depresyon_skoru':[10,10,20,20,30,30],'anksiyete_skoru':[1,1,2,2,3,3],
                'xlsr_meanvar_features':[json.dumps([0.]*2048)]*6})
            frame.to_csv(path,index=False)
            with patch.object(train,'TARGET','anksiyete_skoru'):
                X,y,g,_=train.load(path,'demo',0,'xlsr300m_meanvar')
                self.assertEqual(X.shape,(6,2048));np.testing.assert_array_equal(y,[1,1,2,2,3,3])
                self.assertIn('demo_001',g)
                frame.loc[1,'anksiyete_skoru']=99;frame.to_csv(path,index=False)
                with self.assertRaises(ValueError):train.load(path,'demo',0,'xlsr300m_meanvar')

    def test_fold_holds_out_all_segments_and_uses_median(self):
        X=np.arange(18,dtype=float).reshape(6,3);y=np.array([2,2,2,8,8,8],dtype=float);groups=np.array(['a']*3+['b']*3)
        class FakeModel:
            feature_importances_=np.ones(3)/3
            def fit(self,matrix,target):
                np.testing.assert_array_equal(matrix,X[3:]);np.testing.assert_array_equal(target,y[3:])
            def predict(self,matrix):
                np.testing.assert_array_equal(matrix,X[:3]);return np.array([1.,2.,100.])
        with patch.object(train,'make_model',lambda params:FakeModel()),patch.object(train,'require_gpu_model',lambda model:None):
            result=train.fold(X,y,groups,'a',{},None)
        self.assertEqual(result['predicted_score'],2.)
        self.assertEqual(result['absolute_error'],0.)

    def test_serial_threaded_metrics_and_checkpoint(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(24,16));y=np.repeat([2.,8.,13.,21.],6);groups=np.repeat(['p0','p1','p2','p3'],6)
        params={'learning_rate':.03,'max_depth':4,'n_estimators':20}
        with patch.object(train,'DEVICE','cpu'):
            serial=[train.fold(X,y,groups,pid,params,None) for pid in np.unique(groups)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                parallel=train.lopo(pool,X,y,groups,params);indices=train.select(parallel,5)
                rows=train.lopo(pool,X,y,groups,params,indices)
            np.testing.assert_array_equal([r['predicted_score'] for r in serial],[r['predicted_score'] for r in parallel])
            frame=pd.DataFrame([{k:v for k,v in row.items() if k!='importance'} for row in rows]);metric=train.patient_metrics(frame,np.unique(groups))
            delta=frame.true_score.to_numpy()-frame.predicted_score.to_numpy()
            np.testing.assert_allclose([metric['mae'],metric['rmse']],[np.mean(abs(delta)),np.sqrt(np.mean(delta**2))])
            with self.assertRaises(ValueError):train.patient_metrics(pd.concat([frame,frame.iloc[:1]]))
            with tempfile.TemporaryDirectory() as temp:
                dest=Path(temp);(dest/'identity.json').write_text(json.dumps({'synthetic':True}))
                best=frame.iloc[frame.absolute_error.argmin()];heldout=groups==best.patient_id
                original_fit=XGBRegressor.fit;fits=[]
                def record_fit(model,matrix,target,*args,**kwargs):
                    fits.append((matrix.copy(),target.copy()));return original_fit(model,matrix,target,*args,**kwargs)
                with patch.object(XGBRegressor,'fit',record_fit):train.save_best_fold_model(dest,frame,X,y,groups,params,indices)
                self.assertEqual(len(fits),1);np.testing.assert_array_equal(fits[0][0],X[~heldout][:,indices]);np.testing.assert_array_equal(fits[0][1],y[~heldout])
                payload=joblib.load(dest/'minimum_error_fold_model.joblib')
                self.assertEqual(payload['held_out_patient'],best.patient_id)
                np.testing.assert_allclose(np.median(payload['model'].predict(X[heldout][:,indices])),best.predicted_score,rtol=1e-7,atol=1e-7)
                self.assertTrue(train.checkpoint_valid(dest));self.assertEqual(len(list(dest.glob('*model.joblib'))),1)
                with (dest/'minimum_error_fold_model.joblib').open('ab') as f:f.write(b'changed')
                self.assertFalse(train.checkpoint_valid(dest))

if __name__=='__main__':unittest.main()
