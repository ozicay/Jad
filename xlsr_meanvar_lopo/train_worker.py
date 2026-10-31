"""One Slurm-bound GPU per worker; eight configs, two configs sequentially per worker."""
import argparse,itertools,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--target',required=True,choices=['depresyon_skoru','anksiyete_skoru']);a=p.parse_args()
rank=int(os.environ['SLURM_PROCID']);tasks=int(os.environ['SLURM_NTASKS'])
if tasks!=4:raise ValueError('This example requires exactly four Slurm tasks')
root=Path(__file__).resolve().parent;out=Path(a.output);logdir=out/'logs';logdir.mkdir(parents=True,exist_ok=True)
configs=list(itertools.product(['geriatric','non_geriatric'],['all','5','10','15']));failures=0
for dataset,selection in configs[rank::tasks]:
    with (logdir/f'{dataset}_{selection}.log').open('a') as log:
        command=[sys.executable,str(root/'train_lopo.py'),'--csv',str(Path(a.data_dir)/f'{dataset}.csv'),
                 '--output',str(out),'--target',a.target,'--dataset',dataset,'--selection',selection,
                 '--device','cuda:0','--trials','20','--workers','5']
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        failures+=result.returncode!=0
        print(f'rank={rank} dataset={dataset} selection={selection} returncode={result.returncode}',flush=True)
raise SystemExit(1 if failures else 0)
