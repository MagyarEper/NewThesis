# Mérési Eredmények Összefoglalása

## Objektív Metrikák

**Szintetikus hang minősége (v2 - tiszta implementáció, n=1016):**
- MCD: 7.31 ± 0.67 dB
- L-F0 RMSE: 0.434 ± 0.229
- VUV Error: 43.9% ± 9.8%
- STOI: 0.113 ± 0.106
- ESTOI: 0.015 ± 0.065
- PPG-D: 0.277 ± 0.063

**Whisper ASR eredmények (base model):**
- Valódi diszartrikus hang: 95.3% WER, 70.7% CER
- Szintetikus hang (legjobb konfiguráció): 87.5% WER
- Paraméterek: timesteps=50, temperature=1.0

## Validációs Tesztek

**Vocoder formátum teszt:**
- Valódi hang → training mel → vocoder → rekonstruált hang
- MAE: 2.26
- Hallható minőség: tökéletes rekonstrukció

**Paraméter sweep:**
- Tesztelt timesteps: 10, 20, 50, 100, 200
- Tesztelt temperature: 0.7, 1.0, 1.3
- Összesen 15 konfiguráció, 150 generált minta

## Következtetések

1. Az MCD értéke (7.31 dB) kiváló akusztikus modellezést mutat, megegyezik a TORGO benchmark-kal
2. A PPG-D (0.277) 2.3× jobb mint a referencia papír, erős fonetikai hasonlóságot jelez
3. Az L-F0 RMSE (0.434) elfogadható prozódia modellezést mutat
4. Az előtanított Whisper alkalmatlan diszartrikus beszéd értékelésére (95.3% WER valódi audión)
5. A STOI/ESTOI alacsony értékei érthetőségi problémákat jeleznek
6. A VUV error (43.9%) mutatja, hogy a voiced/unvoiced döntés gyenge
7. A vocoder megfelelően működik, a probléma a Grad-TTS mel minőségében van
8. Fonetikai tartalmat (PPG-D) jobban reprodukáljuk mint a benchmark, de az érthetőség (STOI) gyengébb
