"""Portable version of the established global-importance XLS-R LOPO protocol."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, time, fcntl, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import joblib
from xgboost import XGBRegressor
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import pearsonr
MAX_WORKERS = 5
NUM_TRIALS = 20
TARGET = "depresyon_skoru"
DEVICE = "cuda:0"
PATIENT_COLUMN = "patient_id"

COLS={'xlsr300m_meanvar':'xlsr_meanvar_features'}
DIMS={'xlsr300m_meanvar':2048}

def write_json(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)

def load(root,dataset,duration,feature):
    path=Path(root)
    df=pd.read_csv(path,dtype={PATIENT_COLUMN:str})
    required={PATIENT_COLUMN,TARGET,COLS[feature]}
    if not required.issubset(df): raise ValueError(f'Missing columns: {sorted(required-set(df))}')
    raw=df[PATIENT_COLUMN].fillna('').str.strip().str.lower()
    if raw.isin(['','nan','none']).any(): raise ValueError('Missing patient ID')
    groups=(dataset+'_'+raw).to_numpy()
    y=pd.to_numeric(df[TARGET],errors='raise').to_numpy(dtype=float)
    X=np.vstack([np.asarray(json.loads(v),dtype=float) for v in df[COLS[feature]]])
    if X.shape!=(len(df),DIMS[feature]) or not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('Invalid features/target')
    if (X[:,1024:]<0).any(): raise ValueError('Negative variance')
    if len(np.unique(groups))<3: raise ValueError('At least three patients required')
    for pid in np.unique(groups):
        if len(np.unique(y[groups==pid]))!=1: raise ValueError('Inconsistent patient targets')
    evidence=[{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]
    return X,y,groups,evidence


def make_model(params):
    return XGBRegressor(**params,objective='reg:squarederror',tree_method='hist',device=DEVICE,n_jobs=1,random_state=42)

def fold(X,y,groups,pid,params,indices):
    try:
        test=groups==pid; matrix=X if indices is None else X[:,indices]
        model=make_model(params); model.fit(matrix[~test],y[~test])
        require_gpu_model(model)
        pred=float(np.median(model.predict(matrix[test])))
        return {'patient_id':pid,'true_score':float(y[test][0]),'predicted_score':pred,
                'absolute_error':abs(float(y[test][0])-pred),'squared_error':(float(y[test][0])-pred)**2,
                'importance':model.feature_importances_.copy()}
    except Exception:
        traceback.print_exc(); raise

def lopo(pool,X,y,groups,params,indices=None):
    # executor.map preserves np.unique patient order and propagates exceptions.
    def task(pid): return fold(X,y,groups,pid,params,indices)
    return list(pool.map(task,np.unique(groups)))

def select(rows,k,final=False):
    # Reference final ranking reads importance CSV rounded to six decimals.
    matrix=np.array([r['importance'] for r in rows],dtype=float)
    if final: matrix=np.array([[float(f'{v:.6f}') for v in row] for row in matrix])
    df=pd.DataFrame(matrix,columns=[f'feat_{i}' for i in range(matrix.shape[1])])
    ranked=df.mean().sort_values(ascending=False).head(k).index
    return sorted(int(name.split('_')[1]) for name in ranked)

def require_gpu_model(model):
    actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if DEVICE.startswith('cuda') and not actual.startswith('cuda'): raise RuntimeError(f'GPU requested but actual device={actual}')
    if DEVICE=='cpu' and actual!='cpu': raise RuntimeError(f'CPU requested but actual device={actual}')


def gpu_environment():
    report={'device':DEVICE,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'hostname':os.uname().nodename,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'workers':MAX_WORKERS}
    if DEVICE.startswith('cuda'):
        import ctypes
        cuda=ctypes.CDLL('libcuda.so.1');count=ctypes.c_int()
        if cuda.cuInit(0)!=0 or cuda.cuDeviceGetCount(ctypes.byref(count))!=0 or count.value<1: raise RuntimeError('CUDA unavailable')
        if os.environ.get('SLURM_PROCID') is not None and count.value!=1: raise RuntimeError('Expected one assigned GPU per Slurm task')
        report['cuda_device_count']=count.value
    return report


def patient_metrics(frame,expected=None):
    required={'patient_id','true_score','predicted_score','absolute_error','squared_error'}
    if not required.issubset(frame): raise ValueError('Missing patient columns')
    if frame.patient_id.duplicated().any() or frame.patient_id.astype(str).str.startswith('OVERALL').any():
        raise ValueError('Duplicate patient or summary rows')
    if expected is not None and set(frame.patient_id)!=set(expected): raise ValueError('Patient IDs do not match LOPO patients')
    true=frame.true_score.to_numpy(dtype=float); pred=frame.predicted_score.to_numpy(dtype=float)
    if not len(true) or not np.isfinite(true).all() or not np.isfinite(pred).all(): raise ValueError('Invalid patient scores')
    np.testing.assert_allclose(frame.absolute_error,np.abs(true-pred),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(frame.squared_error,(true-pred)**2,rtol=1e-12,atol=1e-12)
    reason=None
    if len(true)<2: reason='fewer_than_two_patients'
    elif np.ptp(true)==0: reason='constant_true_scores'
    elif np.ptp(pred)==0: reason='constant_predictions'
    if reason: r=p=float('nan')
    else:
        result=pearsonr(true,pred,alternative='two-sided'); r=float(result.statistic); p=float(result.pvalue)
    return {'rmse':float(np.sqrt(np.mean((true-pred)**2))), 'mae':float(np.mean(np.abs(true-pred))),
            'pearson_r':r,'pearson_p':p,'pearson_reason':reason,'num_patients':len(true)}


def atomic_dump(value,path):
    tmp=path.with_suffix('.tmp'); joblib.dump(value,tmp); tmp.replace(path)


def checkpoint_valid(dest):
    try:
        report=json.loads((dest/'best_fold_model.json').read_text())
        return report['status']=='verified_complete' and hashlib.sha256((dest/report['file']).read_bytes()).hexdigest()==report['sha256']
    except (FileNotFoundError,KeyError,ValueError):
        return False


def save_best_fold_model(dest,frame,X,y,groups,params,indices):
    best=frame.iloc[int(frame.absolute_error.to_numpy().argmin())]
    test=groups==best.patient_id
    model=make_model(params); model.fit(X[~test][:,indices],y[~test]); require_gpu_model(model)
    reproduced=float(np.median(model.predict(X[test][:,indices])))
    np.testing.assert_allclose(reproduced,float(best.predicted_score),rtol=1e-7,atol=1e-7)
    name='minimum_error_fold_model.joblib'
    atomic_dump({'model':model,'selected_feature_indices':indices,'best_params':params,
                 'held_out_patient':best.patient_id,'true_score':float(best.true_score),
                 'predicted_score':reproduced,'absolute_error':float(best.absolute_error),
                 'selection':'retrospective_minimum_patient_absolute_error_not_generalization',
                 'identity':json.loads((dest/'identity.json').read_text())},dest/name)
    write_json(dest/'best_fold_model.json',{'status':'verified_complete','criterion':'held_out_patient_absolute_error',
               'absolute_error':float(best.absolute_error),'prediction_reproduced':True,
               'file':name,'sha256':hashlib.sha256((dest/name).read_bytes()).hexdigest()})


def predict_frame(pool,X,y,groups,params,indices,dataset):
    rows=lopo(pool,X,y,groups,params,indices)
    frame=pd.DataFrame([{k:v for k,v in row.items() if k!='importance'} for row in rows])
    frame['dataset']=dataset
    frame['status']='success'
    return frame


def config_name(config):
    d,s,f,k=config
    return f'{d}_{f}_{"all" if k==0 else "top"+str(k)}'


def run(root,out,config,checkpoint=False,rerun=False):
    dataset,duration,feature,k=config; dest=Path(out)/config_name(config); dest.mkdir(parents=True,exist_ok=True)
    with (dest/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        X,y,groups,evidence=load(root,dataset,duration,feature)
        identity={'config':list(config),'inputs':evidence,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'target':TARGET,'patient_column':PATIENT_COLUMN,'n_trials':NUM_TRIALS,'device':DEVICE,'max_workers':MAX_WORKERS}
        if (dest/'identity.json').exists():
            if json.loads((dest/'identity.json').read_text())!=identity: raise ValueError('Resume identity mismatch')
        elif rerun: raise ValueError('Cannot rerun without saved identity')
        else: write_json(dest/'identity.json',identity)
        if (dest/'metrics.json').exists() and not rerun:
            saved=json.loads((dest/'metrics.json').read_text())
            if saved.get('status')=='complete' and (not checkpoint or checkpoint_valid(dest)): return
        gpu=gpu_environment(); write_json(dest/'gpu_allocation.json',gpu)
        if rerun:
            params=json.loads((dest/'best_params.json').read_text())
            indices=json.loads((dest/'selected_features.json').read_text())['selected_feature_indices']
            original=json.loads((dest/'metrics.json').read_text())
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                frame=predict_frame(pool,X,y,groups,params,indices,dataset)
            previous=pd.read_csv(dest/'patient_predictions.csv').set_index('patient_id').sort_index()
            current=frame.set_index('patient_id').sort_index()
            if not previous.index.equals(current.index): raise ValueError('Rerun patient mismatch')
            np.testing.assert_allclose(previous[['true_score','predicted_score']],current[['true_score','predicted_score']],rtol=1e-7,atol=1e-7)
            metrics=patient_metrics(frame,np.unique(groups))
            for key in ('rmse','mae','pearson_r','pearson_p'):
                np.testing.assert_allclose(metrics[key],original[key],rtol=1e-7,atol=1e-7,equal_nan=True)
            frame.to_csv(dest/'rerun_patient_predictions.csv',index=False)
            write_json(dest/'rerun_verification.json',{'passed':True,'rtol':1e-7,'atol':1e-7,'metrics':metrics})
            return
        started=time.time()
        initial={'rmse':None,'mae':None,'pearson_r':None,'pearson_p':None,'num_patients':len(np.unique(groups)),
                 'num_optuna_trials':NUM_TRIALS,'best_params':{},'selected_feature_indices':[],
                 'execution_seconds':None,'status':'running','config':list(config)}
        write_json(dest/'metrics.json',initial)
        try:
            study=optuna.create_study(storage='sqlite:///'+str((dest/'study.sqlite').resolve()),study_name=config_name(config),direction='minimize',load_if_exists=True)
            for trial in study.trials:
                if trial.state==optuna.trial.TrialState.RUNNING: study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                def objective(trial):
                    params={'learning_rate':trial.suggest_float('learning_rate',1e-3,0.1,log=True),
                            'max_depth':trial.suggest_int('max_depth',4,14),
                            'n_estimators':trial.suggest_int('n_estimators',100,500)}
                    full=lopo(pool,X,y,groups,params)
                    indices=select(full,k) if k else None
                    results=lopo(pool,X,y,groups,params,indices) if k else full
                    trial.set_user_attr('selected_feature_indices',indices)
                    patient_mae=float(np.mean([row['absolute_error'] for row in results]))
                    return patient_mae
                complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
                if complete>NUM_TRIALS: raise RuntimeError('Unexpected extra trials')
                study.optimize(objective,n_trials=NUM_TRIALS-complete,n_jobs=1)
                params=study.best_params
                write_json(dest/'best_params.json',params)
                study.trials_dataframe().to_csv(dest/'optuna_trials.csv',index=False)
                checkpoint_path=dest/'final_full.joblib'
                if checkpoint_path.exists():
                    cached=joblib.load(checkpoint_path)
                    if cached['best_params']!=params or cached['identity']!=identity: raise ValueError('Stale final checkpoint')
                    full=cached['rows']
                else:
                    full=lopo(pool,X,y,groups,params)
                    atomic_dump({'best_params':params,'identity':identity,'rows':full},checkpoint_path)
                pd.DataFrame([r['importance'] for r in full],index=[r['patient_id'] for r in full],
                             columns=[f'feat_{i}' for i in range(X.shape[1])]).to_csv(dest/'final_full_importance.csv',float_format='%.6f',index_label='patient_id')
                indices=select(full,k,final=True) if k else list(range(X.shape[1]))
                write_json(dest/'selected_features.json',{'selected_feature_indices':indices,'top_k':k,'source':'final_full_lopo_best_params','importance_precision':6})
                if k: frame=predict_frame(pool,X,y,groups,params,indices,dataset)
                else:
                    # All uses this one final full LOPO; no second LOPO or selection.
                    frame=pd.DataFrame([{key:v for key,v in row.items() if key!='importance'} for row in full])
                    frame['dataset']=dataset
                    frame['status']='success'
            metrics=patient_metrics(frame,np.unique(groups))
            frame.to_csv(dest/'patient_predictions.csv',index=False)
            metrics.update(target=TARGET,dataset=dataset,duration="native",feature=feature,selection='All' if not k else f'Top-{k}',
                           num_optuna_trials=NUM_TRIALS,best_params=params,selected_feature_indices=indices,
                           optuna_objective_mae=study.best_value,execution_seconds=time.time()-started,
                           total_completed_trial_seconds=sum(t.duration.total_seconds() for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE),
                           status='complete',gpu_allocation=gpu,checkpoint_requested=checkpoint)
            write_json(dest/'metrics.json',metrics)
            if checkpoint:
                save_best_fold_model(dest,frame,X,y,groups,params,indices)
        except Exception as exc:
            # Keep completed evaluation valid even if optional model storage fails.
            current=json.loads((dest/'metrics.json').read_text())
            if current.get('status')=='complete': current['checkpoint_error']=str(exc)
            else: current.update(status='failed',error=str(exc),execution_seconds=time.time()-started)
            write_json(dest/'metrics.json',current); traceback.print_exc(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True,help='CSV produced by extract_features.py')
    parser.add_argument('--output',required=True)
    parser.add_argument('--target',choices=['depresyon_skoru','anksiyete_skoru'],required=True)
    parser.add_argument('--dataset',required=True,help='Output label, e.g. geriatric or non_geriatric')
    parser.add_argument('--patient-column',default='patient_id')
    parser.add_argument('--selection',choices=['all','5','10','15'],required=True)
    parser.add_argument('--device',default='cuda:0',help='cuda:0 or cpu')
    parser.add_argument('--trials',type=int,default=20)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--rerun',action='store_true',help='Verify saved parameters/indices without new Optuna trials')
    args=parser.parse_args()
    if args.trials<1 or args.workers<1: parser.error('trials and workers must be positive')
    if args.device!='cpu' and not (args.device.startswith('cuda:') and args.device[5:].isdigit()): parser.error('device must be cpu or cuda:N')
    if not args.dataset or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.dataset): parser.error('dataset may contain letters, digits, underscore and hyphen')
    TARGET=args.target;DEVICE=args.device;NUM_TRIALS=args.trials;MAX_WORKERS=args.workers;PATIENT_COLUMN=args.patient_column
    k=0 if args.selection=='all' else int(args.selection)
    # Target in the path keeps depression/anxiety runs independent.
    run(args.csv,Path(args.output)/TARGET,(args.dataset,0,'xlsr300m_meanvar',k),checkpoint=True,rerun=args.rerun)
