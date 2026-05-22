# Grad-TTS újratanítás – v2 (2026-05-06)

## Motiváció

Az eredeti 500 epochos modell (`logs/hungarian_dysarthria`) csak training loss-t logolt, és a validációs filelist (`valid.txt`) ~41%-os szöveg-átfedést mutatott a train settel. Ez overfitting detektálását lehetetlenné tette.

---

## Változtatások

### 1. Szöveg-alapú train/val split

- **Régi**: véletlenszerű sor-szintű split (szöveg-átfedés lehetséges)
- **Új**: egyedi szövegek 85/15 arányban szétválasztva (`random.seed(42)`)
- Train: `resources/filelists/train_textplit.txt` – 7516 bejegyzés, 5117 egyedi szöveg
- Val: `resources/filelists/valid_textsplit.txt` – 1471 bejegyzés, 903 egyedi szöveg
- **0% szöveg-átfedés** train és val között

### 2. Validációs loop hozzáadva (`train_multi_speaker.py`)

- Minden epoch végén fut egy `torch.no_grad()` validációs kör
- Logolt metrikák: `val_dur`, `val_prior`, `val_diff`
- TensorBoard: `validation/duration_loss`, `validation/prior_loss`, `validation/diffusion_loss`
- `train.log` sorok kiegészítve: `| val_dur = X.XXX | val_prior = X.XXX | val_diff = X.XXX`

### 3. Params frissítés (`params.py`)

| Paraméter | Régi | Új |
|---|---|---|
| `train_filelist_path` | `valid.txt` (régi) | `train_textplit.txt` |
| `valid_filelist_path` | – | `valid_textsplit.txt` |
| `log_dir` | `logs/hungarian_dysarthria` | `logs/hungarian_dysarthria_v2` |
| `n_epochs` | 500 | 200 |
| `save_every` | 1 | 10 |

### 4. Torchaudio backend fix (`data.py`)

- `ta.load(filepath)` → `ta.load(filepath, backend="soundfile")`
- Mindkét osztályban (`TextMelDataset` és `TextMelSpeakerDataset`)
- Ok: a `thesis` venv torchaudio verziója ffmpeg backendre állt alapból, ami deep07-en crashelt

---

## Remote szerver konfiguráció (deep07)

- Projekt: `/home/makais/Thesis/NewThesis/NewThesis/`
- Venv: `/home/makais/venvs/thesis`
- Wav fájlok: `wavs_16lhz/` (a mappa neve tényleg `lhz`, nem `khz`)
- Hiányzó fájlok: 11 db kiszűrve a filelistekből a tréning előtt

---

## Futtatás

```bash
cd /home/makais/Thesis/NewThesis/NewThesis/Grad-TTS
source /home/makais/venvs/thesis/bin/activate
nohup python train_multi_speaker.py > train_stdout.log 2>&1 &
```

Monitoring:
```bash
tail -f logs/hungarian_dysarthria_v2/train.log
```
