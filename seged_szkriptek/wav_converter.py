import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm

# Útvonalak
manifest_path = "/home/arcdeus/Documents/NewThesis/manifest.csv"
output_dir = Path("/home/arcdeus/Documents/NewThesis/wavs_16khz")

# Kimeneti mappa létrehozása
output_dir.mkdir(parents=True, exist_ok=True)

# Manifest beolvasása
df = pd.read_csv(manifest_path)

print(f"Resample-lés 44.1 kHz -> 16 kHz...")
print(f"Összesen {len(df)} fájl feldolgozása")

# Minden wav fájl feldolgozása
for idx, row in tqdm(df.iterrows(), total=len(df)):
    original_wav = row['wav']
    utt_id = row['utt_id']
    
    # Eredeti wav betöltése
    audio, sr = librosa.load(original_wav, sr=None)  # eredeti sr betöltése
    
    # Resample 16 kHz-re
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Új fájl mentése
    output_path = output_dir / f"{utt_id}.wav"
    sf.write(output_path, audio_16k, 16000)

print(f"\n✓ Kész! {len(df)} fájl átkonvertálva.")
print(f"Új fájlok helye: {output_dir}")