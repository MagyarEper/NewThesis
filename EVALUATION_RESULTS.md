# Mérési Eredmények Összefoglalása

## Objektív Metrikák

**Szintetikus hang minősége:**
- MCD: 6.79 dB
- PSTOI: 0.104
- Mel távolság: 75 dB
- Korreláció: 0.338

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

1. Az MCD értéke jó akusztikus modellezést mutat
2. Az előtanított Whisper alkalmatlan diszartrikus beszéd értékelésére
3. A PSTOI alacsony értéke a mel generálás minőségi problémáját jelzi
4. A vocoder megfelelően működik, a probléma a Grad-TTS mel kimenetében van
