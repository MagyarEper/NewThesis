import sys
sys.path.append('/home/arcdeus/Documents/NewThesis/Grad-TTS')

import torch
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from torch.utils.data import DataLoader

# Dataset betöltése
print("Dataset betöltése...")
dataset = TextMelSpeakerDataset(
    filelist_path='/home/arcdeus/Documents/NewThesis/Grad-TTS/resources/filelists/train.txt',
    cmudict_path='/home/arcdeus/Documents/NewThesis/Grad-TTS/resources/cmu_dictionary',
    add_blank=True,
    n_fft=1024,
    n_mels=80,
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    f_min=0.0,
    f_max=8000.0
)

print(f"Dataset méret: {len(dataset)} sample")

# Collate function
collate_fn = TextMelSpeakerBatchCollate()

# DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Első batch betöltése
print("\nElső batch betöltése...")
batch = next(iter(dataloader))

# Ellenőrzések
print("\n" + "="*60)
print("MEL-SPEKTROGRAM ELLENŐRZÉS")
print("="*60)

# Batch struktúra
print(f"\nBatch kulcsok: {batch.keys() if isinstance(batch, dict) else 'Tuple/List'}")

# Ha tuple/list, akkor kicsomagolás
if isinstance(batch, (tuple, list)):
    x, x_lengths, y, y_lengths, spk = batch
    print(f"\nBatch elemek:")
    print(f"  x (text): {x.shape}, dtype: {x.dtype}")
    print(f"  x_lengths: {x_lengths.shape}")
    print(f"  y (mel): {y.shape}, dtype: {y.dtype}")
    print(f"  y_lengths: {y_lengths.shape}")
    print(f"  spk (speaker): {spk.shape}")
    
    mel = y
else:
    mel = batch['y']
    print(f"\nMel shape: {mel.shape}")
    print(f"Mel dtype: {mel.dtype}")

# Shape ellenőrzés
print(f"\n1. SHAPE ellenőrzés:")
print(f"   Mel shape: {mel.shape}")
print(f"   ✓ 80 mel-bins? {80 in mel.shape}")
if 80 in mel.shape:
    mel_dim_idx = list(mel.shape).index(80)
    print(f"   ✓ 80-as dimenzió indexe: {mel_dim_idx}")

# Dtype ellenőrzés
print(f"\n2. DTYPE ellenőrzés:")
print(f"   Dtype: {mel.dtype}")
print(f"   ✓ Float? {mel.dtype in [torch.float32, torch.float64, torch.float16]}")

# Értéktartomány ellenőrzés
print(f"\n3. ÉRTÉKTARTOMÁNY ellenőrzés:")
print(f"   Min: {mel.min().item():.4f}")
print(f"   Max: {mel.max().item():.4f}")
print(f"   Mean: {mel.mean().item():.4f}")
print(f"   Std: {mel.std().item():.4f}")

# NaN/Inf ellenőrzés
has_nan = torch.isnan(mel).any().item()
has_inf = torch.isinf(mel).any().item()
print(f"\n4. NaN/Inf ellenőrzés:")
print(f"   NaN értékek? {has_nan}")
print(f"   Inf értékek? {has_inf}")

# Értékelés
print("\n" + "="*60)
print("ÖSSZEGZÉS:")
print("="*60)

issues = []
if 80 not in mel.shape:
    issues.append("❌ Nincs 80-as dimenzió!")
if mel.dtype not in [torch.float32, torch.float64, torch.float16]:
    issues.append("❌ Nem float dtype!")
if has_nan:
    issues.append("❌ Van NaN érték!")
if has_inf:
    issues.append("❌ Van Inf érték!")
if mel.min().item() < -50 or mel.max().item() > 50:
    issues.append("⚠️  Extrém értékek (< -50 vagy > 50)")

if not issues:
    print("✅ Minden ellenőrzés sikeres!")
    print("✅ A dataset helyesen adja vissza a mel-spektrogramokat!")
else:
    print("❌ Problémák találva:")
    for issue in issues:
        print(f"   {issue}")

# Egy példa mel vizualizáció
print("\n" + "="*60)
print("MEL-SPEKTROGRAM PÉLDA (első sample):")
print("="*60)
print(f"Shape: {mel[0].shape}")
print(f"Első 5x5 érték:")
if mel[0].dim() == 2:
    print(mel[0][:5, :5])
else:
    print(mel[0][:, :5, :5] if mel[0].shape[0] < mel[0].shape[-1] else mel[0][:5, :5])
