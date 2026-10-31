"""Download XLS-R300M once and save a reusable local safetensors checkpoint."""
import argparse
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Model
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--revision');a=p.parse_args()
dest=Path(a.output)
if dest.exists() and any(dest.iterdir()):p.error('Use an empty output directory')
kwargs={'revision':a.revision} if a.revision else {}
model=Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m',**kwargs)
processor=Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-xls-r-300m',**kwargs)
model.save_pretrained(dest,safe_serialization=True);processor.save_pretrained(dest)
print(dest.resolve())
