### Final Model Performance

- **Test Accuracy:** 74.17%
- **Hyperparameters:** lr=0.001, weight_decay=1e-4, batch_size=32
- **Features:** 57 summary stats (including tempo)
- **Augmentation:** Additive Gaussian Noise (std=0.1)

### Confusion Matrix Analysis



- **Strongest Genres:** Disco, Metal, Pop. These have very distinct sonic signatures that are well-captured by the summary features.
- **Weakest Genre:** **Rock (33% accuracy)**. This is the biggest area for improvement.
- **Key Confusions:**
    1. **Rock -> Country/Blues/Disco:** Indicates that the features are too general and cannot distinguish between sub-genres or stylistic overlaps. The model confuses rhythmic drive (Rock/Disco) and guitar tonality (Rock/Country/Blues).
    2. **Hiphop -> Reggae:** Both are rhythm-centric. This prompted the addition of the [[01 - Feature Engineering#tempogram_mean/var|Tempogram feature]].

### Next Steps

Based on this analysis, the primary focus for improvement should be on enhancing the feature set to better capture the unique characteristics of the "Rock" genre.