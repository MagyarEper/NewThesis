# Projekt Roadmap

## Állapot: 2026. január 24.

### ✅ Befejezett (Hétfői prezentációig)
- [x] Training befejezve (500 epoch, converged)
- [x] Sanity check működik
- [x] Optimális inference paraméterek megtalálva
- [x] Alap dokumentáció kész (DOKUMENTACIO.md)
- [x] ~70% elfogadható minőség elérve

### 🎯 Hétfő (2026. január 27.) - Első Prezentáció
**Cél:** Bemutató az eddigi eredményekről

**Prezentálható eredmények:**
- ✅ Multi-speaker dysarthria TTS model (39 beszélő)
- ✅ 10,002 utterance training dataset
- ✅ Konvergált model (500 epoch)
- ✅ Működő inference pipeline
- ✅ Sanity check eredmények (30 WAV fájl)
- ✅ Dokumentált limitációk (CMU dictionary, változó minőség)

**Demonstráció:**
```bash
# Sanity check futtatása
cd Grad-TTS
python sanity_check.py \
  --checkpoint logs/hungarian_dysarthria/grad_500.pt \
  --length-scale 1.0 --temperature 1.2 --timesteps 20 \
  --speakers 0 5 10
```

**Beszélni kell róla:**
- Model architektúra (Grad-TTS + HiFi-GAN)
- Training folyamat (~15 óra, converged)
- Minőségi eredmények: 70% jó, 30% változó
- Korlátok: CMU angol phoneme dictionary
- Következő lépés: Magyar phoneme dictionary

---

## 📋 További Fejlesztések (Hétfő után)

### 1. Magyar Phoneme Dictionary Implementáció
**Prioritás:** MAGAS  
**Becsült idő:** 2-3 nap  
**Elvárható javulás:** 70% → 85-90% minőség

**Lépések:**
1. [ ] Magyar phoneme mapping létrehozása/megtalálása
   - Források kutatása: Hunphone, magyar IPA mappings
   - Saját dictionary építése ha szükséges
   
2. [ ] Kód módosítás inference-hez
   - `text/symbols.py` - magyar phoneme-ek hozzáadása
   - `text/cleaners.py` - magyar text normalizáció
   - `text/cmudict.py` vagy új `hundict.py` létrehozása
   
3. [ ] Tesztelés meglévő checkpointtal
   - Inference futtatása magyar phoneme-ekkel
   - Sanity check összehasonlítás: angol vs magyar dict
   - Minőség értékelés
   
4. [ ] (Opcionális) Újratanítás magyar phoneme-ekkel
   - Ha inference javulás nagy → érdemes újratanítani
   - 500 epoch (~15 óra)
   - Összehasonlítás: angol-CMU model vs magyar model

### 2. Evaluáció (2. Rész)
**Prioritás:** KÖZEPES  
**Becsült idő:** 3-5 nap

**Objektív metrikák:**
- [ ] Mel-Cepstral Distortion (MCD) számítás
- [ ] F0 RMSE (pitch accuracy)
- [ ] Duration accuracy
- [ ] Per-speaker/per-phoneme analysis

**Szubjektív értékelés:**
- [ ] Listening test protokoll
- [ ] MOS (Mean Opinion Score) ha van panel
- [ ] Intelligibility scoring
- [ ] CMU vs Magyar dictionary összehasonlítás

### 3. Thesis Írás
**Prioritás:** FOLYAMATOS  
**Deadline:** TBD

**Fejezetek:**
- [ ] Bevezetés & Related Work
- [ ] Módszertan (model leírás)
- [ ] Implementáció (training details)
- [ ] Eredmények (metrics + analysis)
- [ ] Összegzés & Future Work

---

## 🔬 Opcionális Kísérletek (Ha van idő)

### A. Különböző Checkpointok Tesztelése
- [ ] Epoch 300, 400, 500 összehasonlítása
- [ ] Van-e "sweet spot" epoch szám

### B. Timesteps Sweep
- [ ] 10, 20, 30, 50 timesteps tesztelése
- [ ] Quality vs. speed trade-off dokumentálása

### C. Speaker-Specific Fine-tuning
- [ ] Problémás beszélők azonosítása
- [ ] Fine-tuning csak arra a beszélőre
- [ ] Javulás mérése

### D. Alternative Vocoders
- [ ] HiFi-GAN+ tesztelése
- [ ] WaveGlow vagy más vocoder
- [ ] Vocoder összehasonlítás

---

## 📊 Sikerkritériumok

### Minimum (Hétfő):
- ✅ Működő TTS rendszer
- ✅ Dokumentált eredmények
- ✅ Demonstrálható minták

### Cél (Magyar dict után):
- 🎯 85-90% jó minőségű utterance-ek
- 🎯 Javulás kimutatása metrikákkal
- 🎯 Publikálható eredmény

### Ideális (Ha minden sikerül):
- ⭐ State-of-art magyar dysarthria TTS
- ⭐ Magyar phoneme dictionary contribution
- ⭐ Konferencia/journal publikáció

---

## 🛠️ Technikai TODO

### Infrastruktúra
- [ ] TensorBoard logok archiválása
- [ ] Checkpoint-ok biztonsági mentése
- [ ] Training script cleanup
- [ ] Environment requirements.txt finalizálása

### Dokumentáció
- [x] DOKUMENTACIO.md (Part 1)
- [ ] DOKUMENTACIO.md (Part 2 - Evaluation)
- [ ] Magyar phoneme dictionary dokumentáció
- [ ] API documentation (ha kell)

### Code Quality
- [ ] Type hints hozzáadása
- [ ] Docstring-ek kiegészítése
- [ ] Error handling javítása
- [ ] Unit tesztek (opcionális)

---

**Utolsó frissítés:** 2026. január 24.  
**Következő mérföldkő:** Hétfői prezentáció (2026. január 27.)
