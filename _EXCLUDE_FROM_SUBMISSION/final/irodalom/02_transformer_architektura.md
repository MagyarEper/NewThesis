# Irodalom vázlat: Transformer architektúra

**Beillesztés helye:** `Dolgozat/02_irodalom.md` — 2.3 ASR szekció eleje (Whisper előtt)  
**Méret:** ~200–350 szó, 1 bekezdés + esetleg 1 ábra referencia  
**Prioritás:** KÖZEPES — Whisper Transformer-alapú, de a dolgozat nem magyarázza

---

## Mit kell lefedni

### 2.x. Transformer architektúra és alkalmazása a beszédfeldolgozásban

**Szükséges tartalom:**

1. **Transformer alapötlet (1-2 mondat)**  
   A Transformer architektúrát Vaswani et al. (2017) vezette be: teljesen attention-alapú encoder-decoder, rekurrens rétegek nélkül. A self-attention mechanizmus minden pozícióból minden más pozícióra tud figyelni, ami hosszú távú függőségek tanulását teszi lehetővé.

2. **Encoder-decoder felépítés (1 bekezdés)**  
   - **Encoder:** bemeneti szekvencia → kontextuális reprezentációk  
     - Multi-head self-attention + feed-forward rétegek  
     - Pozíciókódolás (sinusoidal vagy tanult)  
   - **Decoder:** encoder kimenet + korábbi tokenek → következő token  
     - Masked self-attention (autoregresszív generálás)  
     - Cross-attention az encoder kimenetre

3. **Kapcsolat a Whisperrel (1-2 mondat)**  
   A Whisper modell encoder-decoder Transformer: az encoder a log-mel-spektrogram-ot dolgozza fel, a decoder autoregresszív módon szöveges tokeneket generál. A LoRA finomhangolás a Transformer self-attention súlyait módosítja alacsony rangú dekomponálással.

4. **Nem szükséges részletezni:**  
   - Matematikai levezetések  
   - BERT/GPT különbségek  
   - Positional encoding részletei

---

## Kulcsszavak kereséshez

- `Attention is All You Need Vaswani 2017`
- `Transformer encoder decoder speech recognition`
- `Whisper architecture Transformer ASR`

## Ajánlott irodalom (keress rájuk!)

| Forrás | Mit kell megnézni |
|--------|------------------|
| **Vaswani et al. (2017)** — *"Attention Is All You Need", NeurIPS* | **Alapcikk** — az eredeti Transformer; multi-head attention, encoder-decoder felépítés |
| **Dong et al. (2018)** — *"Speech-Transformer", ICASSP* | Az első Transformer-alapú ASR rendszer — jó hivatkozási pont arra, hogy a Transformer speech-re is alkalmazható |
| **Radford et al. (2023)** — *Whisper paper, ICML* | A Whisper architektúrájának leírása: hány réteg, fejek száma, modellméretek (small: 12 réteg, 512 dim) |
| **Hu et al. (2022)** — *"LoRA: Low-Rank Adaptation of LLMs", ICLR* | Ha LoRA-t is itt tárgyalod, ez a forrás — a query/value matricák módosítása |

## Megjegyzés a terjedelemről

Ez nem kell részletes legyen — 1 jól megírt bekezdés elég, ami:
- Megnevezi a Transformer-t és az eredeti papírt
- Elmondja az encoder-decoder struktúrát 2-3 mondatban
- Megmagyarázza, hogy a Whisper erre épül → így érthetővé válik miért működik a LoRA


## Saját szöveg

2.x. A Transformer architektúra és alkalmazása a beszédfeldolgozásban

A modern természetesnyelv-feldolgozás (NLP) és beszédfelismerés (ASR) technológiai alapköve a Vaswani és munkatársai (2017) által bemutatott Transformer architektúra. A korábbi rekurrens (RNN, LSTM) hálózatokkal ellentétben a Transformer teljesen az önfigyelem (self-attention) mechanizmusra épül, elhagyva a szekvenciális feldolgozást. Ez a felépítés lehetővé teszi a bemeneti szekvencia minden eleme közötti párhuzamos műveletvégzést és a hosszú távú függőségek hatékony feltérképezését, ami kritikus fontosságú a beszédjelek időbeli kiterjedése miatt. Bár eredetileg szöveges fordításra tervezték, Dong és munkatársai (2018) úttörő munkája igazolta, hogy a „Speech-Transformer” struktúra kiválóan alkalmazható az akusztikai jelek és a szöveges szekvenciák közötti komplex leképezésekre is.

Az architektúra egy encoder-decoder felépítést követ. Az encoder rétegek a bemeneti szekvenciát (beszéd esetén a spektrogramot) egy magas szintű kontextuális reprezentációvá alakítják multi-head self-attention és előrecsatolt (feed-forward) rétegek segítségével. Mivel a figyelem-mechanizmus önmagában nem érzékeli az elemek sorrendjét, pozíciókódolást (positional encoding) alkalmaznak az időbeli információk megőrzésére. A decoder egység autoregresszív módon generálja a kimeneti tokeneket: a korábban generált tokenekre (masked self-attention) és az encoder által előállított reprezentációra (cross-attention) támaszkodva jósolja meg a következő szöveges elemet.

Ez az alapfelépítés köszön vissza a Whisper modellben is (Radford et al., 2023), ahol az encoder a log-mel-spektrogram jellemzőit dolgozza fel, a decoder pedig a transzkripciót állítja elő. A dolgozat későbbi fejezeteiben tárgyalt LoRA (Low-Rank Adaptation) finomhangolási eljárás (Hu et al., 2022) közvetlenül ezen architektúra sajátosságait használja ki: a finomhangolás során nem a teljes hálózatot, hanem csak a Transformer rétegek attention-mátrixait (jellemzően a Query és Value súlyokat) módosítja alacsony rangú dekomponálással, így érve el hatékony adaptációt minimális paraméterszám-növekedés mellett.
Felhasznált hivatkozások:

    Dong, L., Xu, S., & Xu, B. (2018). Speech-transformer: a no-recurrence sequence-to-sequence model for continuous speech recognition. 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5884-5888.

    Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. International Conference on Learning Representations (ICLR).

    Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. International Conference on Machine Learning (ICML).

    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 30.