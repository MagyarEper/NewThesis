# Irodalom vázlat: Mel-spektrogram és STFT

**Beillesztés helye:** `Dolgozat/02_irodalom.md` — 2.1 vagy 2.2 elejére, önálló alfejezet  
**Méret:** ~300–500 szó + 1 ábra referencia  
**Prioritás:** LEGFONTOSABB — ez a közös kapocs a TTS és ASR pipeline között

---

## Mit kell lefedni

### 2.x. Hangjelreprezentáció: STFT és mel-spektrogram

**Szükséges tartalom:**

1. **Miért nem raw waveform?**  
   Az időtartományos jel közvetlen neurális hálózatos feldolgozása lassú és memóriaigényes; a frekvenciás reprezentáció tömörebb és az emberi hallással jobban megfelel.

2. **Short-Time Fourier Transform (STFT)**  
   - Az egész jel helyett rövid, egymást átfedő ablakokat FFT-zünk  
   - Eredmény: komplex spektrogram → $|STFT|^2$ → hatványspektrogram  
   - Paraméterek: ablakhossz (pl. 1024), hop size (pl. 256), ablakfüggvény (Hann)

3. **Mel-szűrőbank**  
   - Az emberi hallás nem lineárisan érzékeli a frekvenciákat → mel-skála  
   - Mel-szűrőbank: háromszög alakú frekvenciaablakok, mel-skálán egyenletesen elosztva  
   - Tipikusan 80 mel-bin (Whisper) vagy 80 (Grad-TTS)  
   - Eredmény: mel-spektrogram mátrix: $[T \times n_{mel}]$

4. **Log-mel-spektrogram**  
   - Logaritmikus kompresszió a dinamikatartomány csökkentésére  
   - $S_{log} = \log(\max(S_{mel}, \epsilon))$  
   - Ez a Whisper bemeneti reprezentációja

5. **Kapcsolat a pipeline-nal**  
   - Grad-TTS kimenet: mel-spektrogram → HiFi-GAN → WAV  
   - Whisper bemenet: log-mel-spektrogram (30 mp, 80 bin, 100 frame/s)  
   - Tehát minden kísérletben a mel-spektrogram a köztes reprezentáció

---

## Kulcsszavak kereséshez

- `Short-Time Fourier Transform speech processing`
- `mel filterbank speech feature extraction`
- `log mel spectrogram automatic speech recognition`
- `mel spectrogram TTS vocoder`

## Ajánlott irodalom (keress rájuk!)

| Forrás | Mit kell megnézni |
|--------|------------------|
| **Rabiner & Schafer (1978)** — *Digital Processing of Speech Signals* | Az STFT eredeti leírása (klasszikus referencia) |
| **O'Shaughnessy (1987)** — *Speech Communication* | Mel-skála és emberi hallás kapcsolata |
| **Davis & Mermelstein (1980)** — *IEEE TASLP* | Az MFCC (mel-cepstrum) alappapír — a mel-szűrőbank innen jön; hivatkozott klasszikus |
| **Radford et al. (2023)** — *Whisper paper, ICML* | Leírja a Whisper log-mel konfigurációt (80 bin, 25ms ablak, 10ms hop) |
| **Popov et al. (2021)** — *Grad-TTS paper, ICML* | Leírja a mel-spektrogram konfigurációt amit a Grad-TTS használ |
| **Kong et al. (2020)** — *HiFi-GAN paper, NeurIPS* | Mel→WAV konverzió, HiFi-GAN bemenet formátuma |

## Opcionális ábra

Egy egyszerű blokk-diagram: `WAV → STFT → Mel-szűrőbank → Log → [mel-spektrogram mátrix]`  
Mellé: egy példa mel-spektrogram vizualizáció (normál vs. dizartriás hang összehasonlítás — a `spectrograms/` mappában van ilyen!)


## Saját szöveg

2.x. Hangjelreprezentáció: STFT és mel-spektrogram

A digitális beszédfeldolgozás alapvető kihívása a nagyfelbontású, időtartománybeli nyers hullámforma (raw waveform) hatékony reprezentációja. Bár a neurális hálózatok képesek közvetlenül az időtartományos jelen is operálni, ez számítási szempontból rendkívül igényes és memóriaigényes folyamat a jel magas mintavételi gyakorisága miatt. Ehelyett a modern beszédszintézis (TTS) és automatikus beszédfelismerő (ASR) rendszerek frekvenciatartománybeli reprezentációkat alkalmaznak, amelyek nemcsak tömörebbek, de biológiailag is motiváltak, mivel jobban illeszkednek az emberi hallórendszer jellemzőihez.

A frekvenciaanalízis alapköve a rövid idejű Fourier-transzformáció (Short-Time Fourier Transform, STFT). Mivel a beszédjel stacionáriusnak csak rövid időintervallumokon tekinthető, a teljes jel transzformálása helyett a jelet rövid, egymást átfedő ablakokra bontják, és ezeken végzik el a gyors Fourier-transzformációt (FFT) (Rabiner & Schafer, 1978). A folyamat során olyan paraméterek határozzák meg a felbontást, mint az ablakhossz (például 1024 minta), a lépésköz (hop size, pl. 256 minta) és az ablakfüggvény (jellemzően Hann-ablak), amely a széleknél fellépő szakadások elsimítására szolgál. Az STFT eredménye egy komplex számokból álló mátrix, amelynek abszolútérték-négyzete adja meg a hatványspektrogramot ($|STFT|^2$).

Az emberi fül azonban nem lineárisan érzékeli a frekvenciákat; az alacsonyabb tartományokban finomabb, a magasabbakban durvább a felbontóképessége (O'Shaughnessy, 1987). Ezt a pszichoakusztikai jelenséget modellezi a mel-skála. A spektrogramot egy mel-szűrőbankon vezetik keresztül, amely háromszög alakú, a mel-skálán egyenletesen elosztott sávszűrőkből áll (Davis & Mermelstein, 1980). Az így kapott mel-spektrogram jelentősen csökkenti a dimenzionalitást: a modern rendszerekben, mint a Whisper vagy a Grad-TTS, jellemzően 80 mel-csatornát (bin) használnak a spektrális profil leírására.

A mel-spektrogram értékeit rendszerint logaritmikus skálázásnak vetik alá. A log-mel-spektrogram előállítása során az $S_{log} = \log(\max(S_{mel}, \epsilon))$ képletet alkalmazzák, ahol az $\epsilon$ egy apró konstans a numerikus stabilitás érdekében. Erre a kompresszióra azért van szükség, mert az emberi hangerőérzékelés is logaritmikus jellegű, és ez a transzformáció csökkenti a jel dinamikatartományát, segítve a neurális hálózatok konvergenciáját.

A jelen dolgozatban vizsgált pipeline-ban a mel-spektrogram tölti be a központi, összekötő szerepet. A Whisper ASR modell (Radford et al., 2023) bemenetként 80-csatornás log-mel-spektrogramot használ, amelyet 25 ms-os ablakokkal és 10 ms-os lépésközzel számítanak ki. Ezzel szemben a Grad-TTS (Popov et al., 2021) diffúziós alapú szintézismodell kimenete szintén egy mel-spektrogram, amely a beszéd akusztikai jellemzőit kódolja. Ahhoz, hogy ebből ismét hallható hanghullám jöjjön létre, egy vokóderre van szükség; a HiFi-GAN (Kong et al., 2020) neurális vokóder pontosan ezt a mel-spektrogramot alakítja vissza időtartománybeli hullámformává. Ezáltal a mel-spektrogram a közös interfész, amely lehetővé teszi a szintézis és a felismerés folyamatainak egységes kezelését.
Felhasznált hivatkozások (a bibliográfiához):

    Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366.

    Kong, J., Kim, J., & Bae, J. (2020). HiFi-GAN: Generative adversarial networks for high-fidelity synthesis of spectrograms. Advances in Neural Information Processing Systems, 33, 17022-17033.

    O'Shaughnessy, D. (1987). Speech Communication: Human and Machine. Addison-Wesley.

    Popov, V., Vovk, I., Gogoryan, S., Tasadaq, T., & Korshunov, M. (2021). Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech. Proceedings of the 38th International Conference on Machine Learning (ICML).

    Rabiner, L. R., & Schafer, R. W. (1978). Digital Processing of Speech Signals. Prentice-Hall.

    Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. International Conference on Machine Learning (ICML).