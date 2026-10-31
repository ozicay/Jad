"""Frozen XLS-R300M mean+population-variance extraction; optional Slurm row sharding."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import transformers
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Model
from pooling import pool_mean_variance
COLUMN='xlsr_meanvar_features'
MODEL='facebook/wav2vec2-xls-r-300m'

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        return reader.fieldnames,list(reader)

def write_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def loaded_weights_sha256(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--output',required=True,help='New feature CSV')
    parser.add_argument('--audio-root',help='Base directory for relative file_name values; default is CSV directory')
    parser.add_argument('--model',default=MODEL,help='Hugging Face ID or local XLS-R300M checkpoint directory')
    parser.add_argument('--revision',help='Optional pinned Hugging Face commit')
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--cache-dir')
    parser.add_argument('--rank',type=int,default=int(os.environ.get('SLURM_PROCID','0')))
    parser.add_argument('--world-size',type=int,default=int(os.environ.get('SLURM_NTASKS','1')))
    parser.add_argument('--merge-only',action='store_true')
    args=parser.parse_args()
    if args.world_size<1 or not 0<=args.rank<args.world_size:parser.error('Invalid rank/world size')
    source=Path(args.csv).resolve();output=Path(args.output).resolve()
    if source==output:parser.error('output must differ from input CSV')
    fields,rows=read_csv(source)
    if not rows or not {'file_name','patient_id'}.issubset(fields):parser.error('CSV needs rows, file_name and patient_id')
    if COLUMN in fields:parser.error('Input already has XLS-R features')
    output.parent.mkdir(parents=True,exist_ok=True)
    cache=Path(args.cache_dir or (str(output)+'.parts')).resolve();cache.mkdir(parents=True,exist_ok=True)
    audio_root=Path(args.audio_root).resolve() if args.audio_root else source.parent
    request={'input_sha256':sha(source),'audio_root':str(audio_root),'model':args.model,'revision':args.revision,
             'world_size':args.world_size,'extract_sha256':sha(__file__),'pooling_sha256':sha(Path(__file__).with_name('pooling.py'))}
    identity_path=cache/f'identity_rank_{args.rank}.json'
    if not args.merge_only:
        if args.device!='cpu' and not args.device.startswith('cuda:'):parser.error('device must be cpu or cuda:N')
        if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu for CPU')
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        kwargs={'local_files_only':args.local_files_only}
        if args.revision:kwargs['revision']=args.revision
        processor=Wav2Vec2FeatureExtractor.from_pretrained(args.model,**kwargs)
        model=Wav2Vec2Model.from_pretrained(args.model,attn_implementation='eager',**kwargs).float().eval()
        if model.config.hidden_size!=1024 or model.config.num_hidden_layers!=24:raise ValueError('Expected XLS-R300M architecture (1024 hidden,24 layers)')
        if processor.sampling_rate!=16000:raise ValueError('Expected 16kHz processor')
        identity={'request':request,'weights_sha256':loaded_weights_sha256(model),
                  'model_config_sha256':hashlib.sha256(model.config.to_json_string().encode()).hexdigest(),
                  'processor_sha256':hashlib.sha256(processor.to_json_string().encode()).hexdigest(),
                  'resolved_revision':getattr(model.config,'_commit_hash',None),
                  'dimension':2048,'pooling':'concat(time_mean,time_population_variance)',
                  'variance_ddof':0,'frozen':True,'dtype':'float32','sample_rate':16000,
                  'resampling':'scipy.signal.resample_poly default FIR after mono channel mean',
                  'padding':False,'truncation':False,'torch':torch.__version__,'transformers':transformers.__version__,
                  'numpy':np.__version__,'soundfile':sf.__version__}
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Extraction cache identity mismatch; use a fresh cache/output')
        write_json(identity_path,identity)
        model.to(args.device);model.requires_grad_(False)
        rowdir=cache/'rows';rowdir.mkdir(exist_ok=True)
        for index in range(args.rank,len(rows),args.world_size):
            row=rows[index];audio=Path(row['file_name'])
            if not audio.is_absolute():audio=audio_root/audio
            audio_sha=sha(audio);dest=rowdir/f'{index}.npz'
            if dest.exists():
                with np.load(dest,allow_pickle=False) as saved:
                    if str(saved['audio_sha256'])!=audio_sha:raise ValueError('Audio changed since cached inference')
                continue
            waveform,sr=sf.read(audio,dtype='float32',always_2d=True);mono=waveform.mean(axis=1)
            if not np.isfinite(mono).all():raise ValueError('Non-finite audio')
            source_pcm_sha=hashlib.sha256(mono.tobytes()).hexdigest()
            if sr!=16000:
                divisor=math.gcd(sr,16000);mono=resample_poly(mono,16000//divisor,sr//divisor).astype(np.float32)
            if len(mono)<400:raise ValueError('Segment shorter than XLS-R receptive field')
            inputs={k:v.to(args.device) for k,v in processor(mono,sampling_rate=16000,return_tensors='pt',padding=False,truncation=False).items()}
            with torch.inference_mode():
                hidden=model(**inputs).last_hidden_state;vector=pool_mean_variance(hidden).float().cpu().numpy()
                if index==args.rank:
                    reference=hidden[0].float().cpu().numpy().astype(np.float64)
                    np.testing.assert_allclose(vector,np.r_[reference.mean(0),reference.var(0,ddof=0)],rtol=1e-5,atol=1e-6)
                del hidden
            if vector.shape!=(2048,) or not np.isfinite(vector).all() or (vector[1024:]<0).any():raise ValueError('Invalid features')
            temp=dest.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez(stream,vector=vector,audio_path=str(audio.resolve()),audio_sha256=audio_sha,source_pcm_sha256=source_pcm_sha,
                         pcm_sha256=hashlib.sha256(mono.tobytes()).hexdigest())
            temp.replace(dest)
            if index%25==args.rank or index+args.world_size>=len(rows):print(f'rank={args.rank}: row={index+1}/{len(rows)}',flush=True)
        if args.world_size>1:return
    identities=[json.loads((cache/f'identity_rank_{rank}.json').read_text()) for rank in range(args.world_size)]
    if any(item!=identities[0] for item in identities) or identities[0]['request']!=request:raise ValueError('Shard identities differ')
    temp=output.with_suffix(output.suffix+'.partial');pcm_digest=hashlib.sha256()
    if output.exists():
        raise FileExistsError('Output exists; preserve it and choose a new output path')
    with temp.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields+[COLUMN]);writer.writeheader()
        for index,row in enumerate(rows):
            audio=Path(row['file_name'])
            if not audio.is_absolute():audio=audio_root/audio
            with np.load(cache/'rows'/f'{index}.npz',allow_pickle=False) as saved:
                vector=saved['vector'];audio_sha=str(saved['audio_sha256']);pcm_digest.update(str(saved['pcm_sha256']).encode())
                if str(saved['audio_path'])!=str(audio.resolve()) or sha(audio)!=audio_sha:raise ValueError('Audio/row mismatch during merge')
            if vector.shape!=(2048,) or not np.isfinite(vector).all() or (vector[1024:]<0).any():raise ValueError('Invalid cached features')
            writer.writerow({**row,COLUMN:json.dumps(vector.tolist(),allow_nan=False)})
    if sha(source)!=request['input_sha256']:raise ValueError('Source CSV changed during extraction')
    outfields,outrows=read_csv(temp)
    if fields+[COLUMN]!=outfields or len(rows)!=len(outrows):raise ValueError('Output schema/count mismatch')
    if any(any(before[k]!=after[k] for k in fields) for before,after in zip(rows,outrows)):raise ValueError('Metadata changed')
    temp.replace(output)
    write_json(output.with_suffix(output.suffix+'.report.json'),{'status':'verified_complete','identity':identities[0],
               'rows':len(rows),'output_sha256':sha(output),'ordered_pcm_hashes_sha256':pcm_digest.hexdigest()})
    print(f'VERIFIED_COMPLETE: {output}, {len(rows)} rows,2048 features',flush=True)

if __name__=='__main__':main()
