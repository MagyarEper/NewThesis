# Grad-TTS Pipeline Teljes Átvizsgálás

**Dátum:** 2026-05-11  
**Cél:** Megerősíteni, hogy a TTS pipeline technikailag korrekt-e, és a gyenge minőség oka valóban az adathiány.

---

## 1. Vizsgált komponensek

### params.py
| Paraméter | Érték | Helyes? |
|-----------|-------|---------|
| `sample_rate` | 16000 Hz | ✓ |
| `n_spks` | 39 | ✓ |
| `n_feats` | 80 (mel bins) | ✓ |
| `n_fft` | 1024 | ✓ |
| `hop_length` | 256 | ✓ |
| `win_length` | 1024 | ✓ |
| `f_min / f_max` | 0 / 8000 | ✓ |
| `n_enc_channels` | 128 (csökkentett) | ✓ |
| `dec_dim` | 48 (csökkentett) | ✓ |
| `n_epochs` | 120 | ✓ (de kevés) |
| `batch_size` | 32 | ✓ |

### data.py — mel-spektrogram
- SpeechBrain `mel_spectogram` függvény azonos paraméterekkel mint a vocoder
- **Tényleges output shape: `[80, T]`** (empirikusan ellenőrizve)
- `squeeze(0)` → no-op (nincs batch dim), collate helyesen `[B, 80, T]`-t csinál belőle
- Paraméterek: `power=1, normalized=False, min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True`
- **Következtetés: mel pipeline korrekt, kompatibilis a vocoderrel** ✓

### text/ — szöveg előkészítés
- `cleaners.py` → `basic_cleaners`: csak `lowercase` + `collapse_whitespace`, NEM hívja `convert_to_ascii()`-t
- `symbols.py` → grapheme-based, tartalmazza: `á é í ó ö ő ú ü ű Á É Í Ó Ö Ő Ú Ü Ű`
- `text_to_sequence(text, cleaner_names=['basic_cleaners'], dictionary=None)` → magyar karakterek megmaradnak
- **Következtetés: g2p pipeline korrekt, nincs karakter-veszteség** ✓

### train_multi_speaker.py
- Standard Grad-TTS training loop
- Gradient clipping: `max_norm=1` encoder + decoder külön
- Optimizer: Adam, `lr=1e-4`
- Validation loop minden epoch végén
- **Következtetés: training kód korrekt** ✓

### generate_test_set.py (a kísérletekben TÉNYLEGESEN használt szkript)
- `text_to_sequence(text, cleaner_names=['basic_cleaners'], dictionary=None)` → helyes
- `y_dec` shape GradTTS outputból: `[1, 80, T]` (empirikusan ellenőrizve)
- `vocoder.decode_batch(y_dec)` → elvárja `[B, 80, T]` formátumot ✓
- **Következtetés: generálási pipeline korrekt** ✓

### HiFi-GAN vocoder
- `decode_batch(spectrogram)` → elfogad `[B, 80, T]` vagy `[80, T]` formátumot
- `[B, T, 80]` formátummal HIBÁT ad → nincs ilyen eset a kísérletekben ✓
- **Mel kompatibilitás:** data.py és vocoder.py azonos SpeechBrain mel paramétereket használ ✓

---

## 2. Talált mellékhibák (nem a kísérleteket érintik)

| Szkript | Hiba | Hatás |
|---------|------|-------|
| `inference.py` | `text_to_sequence(text, dictionary=cmu)` → `english_cleaners` → `convert_to_ascii()` → `á→a` stb. | Csak manuális tesztre használt, kísérletekre nincs hatás |
| `sanity_check.py` | Ugyanaz mint fent | Csak manuális tesztre használt |
| `sweep_generation_params.py` | Ugyanaz mint fent | Sweep kísérleteket vizsgál, DAug pipeline-t nem érinti |
| `data.py` | Komment `[1, T, 80]`-at ír, valójában `[80, T]` az output | Csak kód-komment hibás, futás helyes |

---

## 3. Adatmennyiség elemzés

| Mutató | Érték |
|--------|-------|
| Tréning fájlok száma | 7 523 |
| Validációs fájlok száma | 1 473 |
| **Összes tréning adat** | **7.78 óra** |
| Hangszók száma | 39 |
| **Átlag/hangszó** | **~12 perc/hangszó** |
| LJSpeech (1 hangszós referencia) | 24 óra |
| Tipikus multi-speaker TTS minimum | 1–5 óra/hangszó |
| Felvételek típusa | Dysarthriás (atipikus akusztika) |

**Tréning lépések:** 7523 / 32 batch ≈ 235 step/epoch × 120 epoch = **~28 200 lépés**  
Összehasonlítás: LJSpeech Grad-TTS tréning tipikusan **1 000 000+ lépés** az eredeti cikkben.

---

## 4. Következtetés

**A Grad-TTS pipeline technikailag korrekt.** A mel-spektrogram számítás, a szöveg előkészítés, a tréning loop és a generálási pipeline mind hibátlan.

A gyenge TTS minőség (STOI=0.113, ~95-97% WER szintetikus hangon) egyértelműen az elégtelen tréning adatból fakad:
- ~12 perc/hangszó a minimálisan szükséges ~1-5 óra helyett
- Dysarthriás, atipikus akusztikájú adat — ehhez várhatóan még több adat kellene
- Mindössze ~28 200 tréning lépés a szokásos 1M+ helyett

**Ez megalapozott dolgozati tanulság:** a dysarthriás TTS scratchből nehéz, az adathiány a szűk keresztmetszet.

---

## 5. Lehetséges javítások (új hangok bevonása nélkül)

Lásd: [TTS_IMPROVEMENT_OPTIONS.md](TTS_IMPROVEMENT_OPTIONS.md)
