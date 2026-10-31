"""Create fake tones for the example CSV. Contains no patient speech."""
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
root=Path(__file__).resolve().parent;out=root/'synthetic';out.mkdir(exist_ok=True)
with (root/'metadata_example.csv').open(newline='') as f:rows=list(csv.DictReader(f))
for index,row in enumerate(rows):
    sr=48000 if index%2 else 16000
    t=np.arange(sr,dtype=np.float32)/sr
    signal=(0.1*np.sin(2*np.pi*(170+index*25)*t)).astype(np.float32)
    sf.write(out/row['file_name'],signal,sr,subtype='FLOAT')
print(out)
