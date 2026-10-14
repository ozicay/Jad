from pathlib import Path
import importlib.metadata
import json
import os
import platform
import sys
import numpy,pandas,sklearn,xgboost,optuna,joblib
modules=[numpy,pandas,sklearn,xgboost,optuna,joblib]
expected={'numpy':'1.26.4','pandas':'2.2.2','sklearn':'1.8.0','xgboost':'3.1.2','optuna':'4.8.0','joblib':'1.4.2'}
for module in modules:
    if module.__version__ != expected[module.__name__]:
        raise RuntimeError(f'{module.__name__}: expected {expected[module.__name__]}, got {module.__version__}')
record={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'packages':{m.__name__:{'version':m.__version__,'path':m.__file__} for m in modules},
        'installed_distributions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record['packages'],indent=2))
