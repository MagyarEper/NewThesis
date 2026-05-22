# Grad-TTS újratanítás – v3 (2026-05-06)

## Motiváció

A v2 modell (`logs/hungarian_dysarthria_v2`) az angol CMU kiejtési szótárat használta tokenizálásra. Ez két problémát okozott:

1. **Magyar ékezetes betűk elvesztek**: az `english_cleaners` cleaner az `unidecode`-ot futtatja, ami `á→a`, `é→e`, `ő→o`, `ű→u` stb. konverziót végez. A training adat tehát ékezet nélkül volt kódolva.
2. **Véletlenszerű angol fonemizálás**: a CMU szótár olyan hétköznapi magyar szavakat mint `"be"`, `"is"`, `"a"` angolul ejtve kódolt (`@B@IY1`, `@IH1@Z`, `@EY1`), mert véletlenül szerepeltek az angol szótárban.

Következmény: a modell vegyes ARPAbet+latin tokeneken tanult, ami out-of-domain szövegeken teljesen érthetetlen hangot produkált (WER > 4.0).

---

## Változtatások

### 1. `Grad-TTS/text/symbols.py` — Magyar grapheme vocab

- **Régi**: 148 token (ASCII + 86 ARPAbet `@`-jelzéssel)
- **Új**: 82 token (ASCII + 18 magyar ékezetes betű, ARPAbet teljesen kivéve)

```
_hungarian = 'áéíóöőúüűÁÉÍÓÖŐÚÜŰ'
```

### 2. `Grad-TTS/data.py` — CMU szótár kikapcsolva

- `from text import text_to_sequence, cmudict` → `from text import text_to_sequence`
- `self.cmudict = cmudict.CMUDict(cmudict_path)` → eltávolítva mindkét osztályból
- `text_to_sequence(text, dictionary=self.cmudict)` → `text_to_sequence(text, cleaner_names=['basic_cleaners'], dictionary=None)`
- `cmudict_path` paraméter eltávolítva a konstruktorokból

### 3. `Grad-TTS/train_multi_speaker.py` — cmudict_path eltávolítva

- `cmudict_path = params.cmudict_path` sor törölve
- `TextMelSpeakerDataset(filelist, cmudict_path, ...)` → `TextMelSpeakerDataset(filelist, ...)`

### 4. `Grad-TTS/params.py` — v3 paraméterek

| Paraméter | v2 | v3 | Indok |
|---|---|---|---|
| `log_dir` | `hungarian_dysarthria_v2` | `hungarian_dysarthria_v3` | Új futtatás |
| `n_epochs` | 200 | 120 | v2-ben 120 epochnál már konvergált |
| `batch_size` | 12 | 32 | 16GB GPU-n ~2.5x gyorsabb epoch |
| `save_every` | 10 | 20 | 120 epochhoz arányos, kevesebb I/O |
| `cmudict_path` | jelen | **eltávolítva** | Nincs többé szükség rá |

### 5. `generate_test_set.py` — CMU szótár kikapcsolva

- `from text import text_to_sequence, cmudict` → `from text import text_to_sequence`
- `cmu = cmudict.CMUDict(...)` sor törölve a `synthesize_utterance` függvényből
- `text_to_sequence(text, dictionary=cmu)` → `text_to_sequence(text, cleaner_names=['basic_cleaners'], dictionary=None)`

---

## Teljesítménybecslés

| | v2 | v3 |
|---|---|---|
| Batch size | 12 | 32 |
| Epochok | 200 | 120 |
| Lépések/epoch | ~626 | ~235 |
| Összes lépés | ~125 200 | ~28 200 |
| Becsült training idő | ~2 nap | **~12 óra** |

*Feltételezve hogy v2 ~2 nap volt deep07-en (A100/RTX GPU, 16GB VRAM)*

---

## Remote szerver (deep07) — futtatás

```bash
cd /home/makais/Thesis/NewThesis/NewThesis/Grad-TTS
source /home/makais/venvs/thesis/bin/activate
nohup python3 train_multi_speaker.py > train_stdout_v3.log 2>&1 &
```

> **Megjegyzés:** `python3` szükséges (nem `python`). A wav fájlok a `wavs_16lhz/` mappában vannak (elírás, nem `wavs_16khz/`) — a filelistek már tartalmazzák a helyes útvonalakat, soha ne írja felül `git pull`.

Monitoring:
```bash
tail -f logs/hungarian_dysarthria_v3/train.log
```

---

## Várható javulás

- Training domain szövegek (smart home parancsok): **érezhetően jobb** érthetőség — a tokenizálás pontosabb, az ékezetes betűk megjelennek a modell bemenetén
- Out-of-domain Wikipedia szövegek: **valamivel jobb**, de a training adat domainkorlátja megmarad — ez kutatási korlátként dokumentálható
