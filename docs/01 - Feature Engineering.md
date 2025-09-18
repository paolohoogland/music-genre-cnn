### Data Source: GTZAN Dataset 

- **Origin:** The GTZAN dataset is used for music genre classification. 
- **Link:** [Kaggle GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) 
- **Contents:** 1000 (30 seconds long) audio tracks.
- **Genres (10):** `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`. 
- **Format:** `.wav` files.

This project uses a set of 59 summary features extracted from each 30-second audio clip using the `librosa` library. The logic for this is primarily in `Audio.py`.

The goal of this is to provide a complex numerical representation of the audio's characteristics for the [[02 - MLP Model Architecture]].

### Feature List

| Feature Name                  | Description                               | Why it's useful                                                                                           |
| :---------------------------- | :---------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `chroma_stft_mean/var`        | Represents the 12 pitch classes.          | Captures harmonic content (chords, melody).                                                               |
| `rms_mean/var`                | Root Mean Square energy.                  | Represents loudness.                                                                                      |
| `spectral_centroid_mean/var`  | The "center of mass" of the spectrum.     | Correlates with the "brightness" of a sound.                                                              |
| `spectral_bandwidth_mean/var` | The width of the spectral band.           | Measures tone variety                                                                                     |
| `rolloff_mean/var`            | Frequency below which a % of energy lies. | Another measure of spectral shape/brightness.                                                             |
| `zero_crossing_rate_mean/var` | Rate at which the signal crosses zero.    | Correlates with noisiness and percussive sounds.                                                          |
| `harmony_mean/var`            | Energy of the harmonic component.         | Measures the strength of pitched/melodic elements.                                                        |
| `perceptr_mean/var`           | Energy of the percussive component.       | Measures the strength of rhythmic elements.                                                               |
| `tempo`                       | The estimated BPM of the track.           | A primary differentiator for genres.                                                                      |
| `mfcc1-20_mean/var`           | Mel-Frequency Cepstral Coefficients.      | Standard features that model human hearing and capture timbre.                                            |
| `tempogram_mean/var`          | Describes rhythmic structure over time.   | Added to solve confusion between genres like Rock, Hiphop, and Reggae. See [[03 - Results and Analysis]]. |