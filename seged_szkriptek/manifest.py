import re
import pandas as pd
from pathlib import Path

xlsx_path = "/home/arcdeus/Documents/Thesis/HungarianDysartriaDatabaseNew/dsyarthria-hun_transcripts.xlsx"
output_txt = "/home/arcdeus/Documents/NewThesis/manifest.txt"

wav_base_dir = "/home/arcdeus/Documents/NewThesis/wavs_16khz"

df = pd.read_excel(xlsx_path)

def remove_brackets(text):
    if pd.isna(text):
        return text
    return re.sub(r'\[|\]', '', text)

# Speaker ID mapping: string -> int
speakers = sorted(df['ID'].unique())
speaker_map = {spk: idx for idx, spk in enumerate(speakers)}

print(f"Speaker mapping:")
for spk, idx in sorted(speaker_map.items())[:5]:
    print(f"  {spk} -> {idx}")
print(f"  ... összesen {len(speaker_map)} speaker")

# Grad-TTS formátum: wav|text|speaker_id (numerikus)
with open(output_txt, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        wav_path = Path(wav_base_dir) / f"{row['Full_ID']}.wav"
        text = remove_brackets(row['Transcript'])
        speaker_id = speaker_map[row['ID']]  # string -> int
        
        # wav|text|speaker_id formátum
        line = f"{wav_path}|{text}|{speaker_id}\n"
        f.write(line)

print(f"\n✓ Manifest létrehozva: {output_txt}")
print(f"✓ Sorok száma: {len(df)}")
print("\nPélda (első 3 sor):")
with open(output_txt, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 3:
            print(f"  {line.strip()}")
        else:
            break
