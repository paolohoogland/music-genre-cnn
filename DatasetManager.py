import os
import random
from collections import defaultdict
import pandas as pd
import re

from Audio import Audio

class DatasetManager:
    OWN_LONG_FEATURES_PATH = 'Data/own_features_30_sec.csv'
    # SHORT_FEATURES_PATH = 'Data/own_features_3_sec.csv'

    def __init__(self):
        # Initialize variables
        self.own_long_features = None
        self.all_genres_list = []

    def extract_audio_features(self):
        audio_instance = Audio()
        audio_path = "./Data/genres_original"
        all_features_list = []

        print("Starting audio feature extraction...")

        try:
            for root, dirs, files in os.walk(audio_path):
                for filename in files:
                    if filename.endswith('.wav'):
                        file_path = os.path.join(root, filename)
                        
                        y, sr = audio_instance.load_audio(file_path)
                        
                        features = {'filename': filename}

                        features['chroma_stft_mean'], features['chroma_stft_var'] = audio_instance.extract_chroma_sftf(y, sr)
                        features['rms_mean'], features['rms_var'] = audio_instance.extract_rms(y)
                        features['spectral_centroid_mean'], features['spectral_centroid_var'] = audio_instance.extract_spectral_centroid(y, sr)
                        features['spectral_bandwidth_mean'], features['spectral_bandwidth_var'] = audio_instance.extract_spectral_bandwith(y, sr)
                        features['rolloff_mean'], features['rolloff_var'] = audio_instance.extract_rolloff(y, sr)
                        features['zero_crossing_rate_mean'], features['zero_crossing_rate_var'] = audio_instance.extract_zero_crossing_rate(y)
                        features['harmony_mean'], features['harmony_var'] = audio_instance.extract_harmony(y, sr)
                        features['perceptr_mean'], features['perceptr_var'] = audio_instance.extract_perceptr(y, sr)
                        features['tempo'] = audio_instance.extract_tempo(y, sr)

                        # Loop through the 20 MFCCs
                        mfcc_means, mfcc_vars = audio_instance.extract_mfccs(y, sr)
                        for i in range(20):
                            features[f'mfcc{i+1}_mean'] = mfcc_means[i]
                            features[f'mfcc{i+1}_var'] = mfcc_vars[i]

                        features['label'] = os.path.basename(root)

                        all_features_list.append(features)
                        print(f"Successfully processed: {filename}")

                        break

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        features_df = pd.DataFrame(all_features_list)
        print(features_df.head())

        return features_df

    def create_feature_dataset(self):
        try: 
            features_df = self.extract_audio_features()
            features_df.to_csv(self.OWN_LONG_FEATURES_PATH, index=False)

            self.own_long_features = features_df
            self.all_genres_list = features_df['label'].unique().tolist()

        except Exception as e:
            print(f"Error processing audio features: {e}")

        print(f"Extracted {len(self.all_genres_list)} unique genres from audio features.")

        print("\n--- Feature Extraction Complete ---")
        print("Head of the new DataFrame:")
        print(features_df.head())

    def get_feature_dataset(self):
        self.own_long_features = self._load_features(self.OWN_LONG_FEATURES_PATH)
        self.all_genres_list = self.get_genre_list()

    def _load_features(self, path):
        # af = audio features, 30 seconds snippet
        af = pd.read_csv(path)

        # af = af.drop(columns=['label'])
        af = af.set_index('filename')

        return af
    
    def get_genre_list(self):
        genres = set(self.own_long_features.index.str.split('.').str[0])
        return sorted(genres)

    def create_sets(self):
        all_data = self.get_all_data_files()

        # Group files by genre
        files_by_genre = defaultdict(lambda: defaultdict(list))

        for path, features, genre in all_data:
            base_id = '.'.join(path.split('.')[:2]) 
            files_by_genre[genre][base_id].append((path, features, genre))

        train_set = []
        val_set = []
        test_set = []
        
        # Fixed seed
        random.seed(42)

        for genre, songs_dict in files_by_genre.items():
            song_ids = list(songs_dict.keys())
            random.shuffle(song_ids)

            num_songs = len(song_ids)
            train_end = int(0.7 * num_songs)
            val_end = train_end + int(0.15 * num_songs)

            train_ids = song_ids[:train_end]
            val_ids = song_ids[train_end:val_end]
            test_ids = song_ids[val_end:]

            # Add all chunks from the same song into the same set
            for sid in train_ids:
                train_set.extend(songs_dict[sid])
            for sid in val_ids:
                val_set.extend(songs_dict[sid])
            for sid in test_ids:
                test_set.extend(songs_dict[sid])

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        
        print(f"Total spectrograms: {len(all_data)}")
        print(f"Training set size: {len(train_set)} images")
        print(f"Validation set size: {len(val_set)} images")
        print(f"Test set size: {len(test_set)} images")

        return train_set, val_set, test_set