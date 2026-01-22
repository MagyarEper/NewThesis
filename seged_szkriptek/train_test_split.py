import pandas as pd
from sklearn.model_selection import train_test_split

# Beolvassuk a manifest.txt-t (pipe-separated: wav|text|speaker_id)
manifest_path = "/home/arcdeus/Documents/NewThesis/manifest.txt"

# Beolvasás: wav|text|speaker_id (már numerikus)
data = []
with open(manifest_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) == 3:
            data.append({'wav': parts[0], 'text': parts[1], 'speaker': int(parts[2])})

df = pd.DataFrame(data)

train_list = []
val_list = []
test_list = []

for speaker in sorted(df['speaker'].unique()):
    speaker_df = df[df['speaker'] == speaker]
    speaker_df = speaker_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_utterances = len(speaker_df)

    if n_utterances < 20:
        train_list.append(speaker_df)
    else:
        train_data, temp_data = train_test_split(speaker_df, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        train_list.append(train_data)
        val_list.append(val_data)
        test_list.append(test_data)

train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True) if val_list else pd.DataFrame(columns=df.columns)
test_df = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame(columns=df.columns)

# Mentés Grad-TTS formátumban (wav|text|speaker)
def save_grad_format(df, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['wav']}|{row['text']}|{row['speaker']}\n")

save_grad_format(train_df, "/home/arcdeus/Documents/NewThesis/Grad-TTS/resources/filelists/libri-tts/train.txt")
save_grad_format(val_df, "/home/arcdeus/Documents/NewThesis/Grad-TTS/resources/filelists/libri-tts/valid.txt")
save_grad_format(test_df, "/home/arcdeus/Documents/NewThesis/Grad-TTS/resources/filelists/libri-tts/test.txt")

print(f"Train: {len(train_df)} utterances")
print(f"Val:   {len(val_df)} utterances")
print(f"Test:  {len(test_df)} utterances")
print(f"\nBeszélők száma train-ben: {train_df['speaker'].nunique()}")
print(f"Beszélők száma val-ban: {val_df['speaker'].nunique()}")
print(f"Beszélők száma test-ben: {test_df['speaker'].nunique()}")