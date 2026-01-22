import sys
from pathlib import Path

# Manifest beolvasása (pipe-separated: wav|text|speaker_id)
manifest_path = "/home/arcdeus/Documents/NewThesis/manifest.txt"

data = []
with open(manifest_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) == 3:
            data.append({'wav': parts[0], 'text': parts[1], 'speaker': parts[2]})

print(f"Összesen {len(data)} fájl a manifestben\n")

# Ellenőrzés
missing = []
existing = []

for item in data:
    wav_path = Path(item['wav'])
    if wav_path.exists():
        existing.append(item)
    else:
        missing.append(item)

# Eredmények
print(f"✓ Megtalált fájlok: {len(existing)}")
print(f"✗ Hiányzó fájlok: {len(missing)}\n")

# Hibaellenőrzés: ha 50-nél több fájl hiányzik
if len(missing) > 50:
    print("❌ HIBA: Több mint 50 fájl hiányzik!")
    print(f"   Hiányzó fájlok száma: {len(missing)}")
    print("\nHiányzó fájlok listája (első 50):")
    for i, item in enumerate(missing[:50], 1):
        print(f"  {i}. {item['wav']}")
    if len(missing) > 50:
        print(f"  ... és még {len(missing) - 50} további")
    sys.exit(1)

if missing:
    print("Hiányzó fájlok listája:")
    for i, item in enumerate(missing, 1):
        print(f"  {i}. {item['wav']}")
    
    # Hiányzó fájlok eltávolítása
    print(f"\n🔧 Hiányzó fájlok eltávolítása a manifestből...")
    
    # Mentés
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for item in existing:
            f.write(f"{item['wav']}|{item['text']}|{item['speaker']}\n")
    
    print(f"✓ Manifest frissítve!")
    print(f"  Eredeti: {len(data)} fájl")
    print(f"  Tisztított: {len(existing)} fájl")
    print(f"  Eltávolítva: {len(missing)} fájl")
else:
    print("🎉 Minden fájl megvan! Nincs mit eltávolítani.")

# Statisztikák speakerenként
print("\n" + "="*50)
print("Statisztikák beszélőnként:")
print("="*50)

# Újraolvasás
final_data = []
with open(manifest_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) == 3:
            final_data.append({'speaker': parts[2]})

# Beszélők számlálása
from collections import Counter
speaker_counts = Counter([item['speaker'] for item in final_data])
for speaker, count in sorted(speaker_counts.items()):
    print(f"C_{speaker.zfill(3)}: {count} fájl")
