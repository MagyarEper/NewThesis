import pandas as pd
from pathlib import Path
import sys

# Manifest beolvasása
manifest_path = "/home/arcdeus/Documents/NewThesis/manifest.csv"
df = pd.read_csv(manifest_path)

print(f"Összesen {len(df)} fájl a manifestben\n")

# Ellenőrzés
missing_indices = []
existing_indices = []

for idx, row in df.iterrows():
    wav_path = Path(row['wav'])
    if wav_path.exists():
        existing_indices.append(idx)
    else:
        missing_indices.append(idx)

# Eredmények
print(f"✓ Megtalált fájlok: {len(existing_indices)}")
print(f"✗ Hiányzó fájlok: {len(missing_indices)}\n")

# Hibaellenőrzés: ha 50-nél több fájl hiányzik
if len(missing_indices) > 50:
    print("❌ HIBA: Több mint 50 fájl hiányzik!")
    print(f"   Hiányzó fájlok száma: {len(missing_indices)}")
    print("\nHiányzó fájlok listája (első 50):")
    for i, idx in enumerate(missing_indices[:50], 1):
        print(f"  {i}. {df.loc[idx, 'wav']}")
    if len(missing_indices) > 50:
        print(f"  ... és még {len(missing_indices) - 50} további")
    sys.exit(1)

if missing_indices:
    print("Hiányzó fájlok listája:")
    for i, idx in enumerate(missing_indices, 1):
        print(f"  {i}. {df.loc[idx, 'wav']}")
    
    # Hiányzó fájlok eltávolítása
    print(f"\n🔧 Hiányzó fájlok eltávolítása a manifestből...")
    df_cleaned = df.loc[existing_indices].reset_index(drop=True)
    
    # Mentés
    df_cleaned.to_csv(manifest_path, index=False, encoding='utf-8')
    print(f"✓ Manifest frissítve!")
    print(f"  Eredeti: {len(df)} fájl")
    print(f"  Tisztított: {len(df_cleaned)} fájl")
    print(f"  Eltávolítva: {len(missing_indices)} fájl")
else:
    print("🎉 Minden fájl megvan! Nincs mit eltávolítani.")

# Statisztikák speakerenként
print("\n" + "="*50)
print("Statisztikák beszélőnként:")
print("="*50)
df_final = pd.read_csv(manifest_path)
speaker_stats = df_final.groupby('speaker').size()
for speaker, count in speaker_stats.items():
    print(f"{speaker}: {count} fájl")
