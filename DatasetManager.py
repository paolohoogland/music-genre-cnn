from fileinput import filename
import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Audio import Audio

class DatasetManager:
    OWN_LONG_FEATURES_PATH = 'Data/own_features_30_sec.csv'
    # SHORT_FEATURES_PATH = 'Data/own_features_3_sec.csv'

    def __init__(self):
        # Initialize variables
        self.own_long_features = None
        self.all_genres_list = []

    def load_features(self, path):
        af = pd.read_csv(path)
        # Get the genres list before dropping the label column
        self.all_genres_list = sorted(af['label'].unique().tolist())
        af = af.drop(columns=['label'])

        # Set the filename as index : allows easy access to rows by filename
        af = af.set_index('filename')
        return af

    def extract_audio_features(self):
        audio_instance = Audio()
        audio_path = "./Data/genres_original"
        all_features_list = []

        print("Starting audio feature extraction...")

        for root, dirs, files in os.walk(audio_path):
            for filename in files:
                if filename.endswith('.wav'):
                    file_path = os.path.join(root, filename)
                    
                    try: 
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
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

        features_df = pd.DataFrame(all_features_list)
        print(features_df.head())

        return features_df

    # def create_feature_dataset(self):
    #     # delete existing csv
    #     if os.path.exists(self.OWN_LONG_FEATURES_PATH):
    #         os.remove(self.OWN_LONG_FEATURES_PATH)

    #     try: 
    #         features_df = self.extract_audio_features()
    #         features_df.to_csv(self.OWN_LONG_FEATURES_PATH, index=False)

    #         self.all_genres_list = features_df['label'].unique().tolist()

    #     except Exception as e:
    #         print(f"Error processing audio features: {e}")

    #     print(f"Extracted {len(self.all_genres_list)} unique genres from audio features.")

    #     print("\n--- Feature Extraction Complete ---")
    #     print("Head of the new DataFrame:")
    #     print(features_df.head())

    def get_feature_dataset(self):
        try:
            df = pd.read_csv(self.OWN_LONG_FEATURES_PATH)
            self.all_genres_list = self.get_genre_list(df)
            self.own_long_features = df.set_index('filename')
        except FileNotFoundError:
            print(f"File not found: {self.OWN_LONG_FEATURES_PATH}")
            return
        except Exception as e:
            print(f"Error processing audio features: {e}")
            return

    def get_genre_list(self, df):
        if 'label' in df.columns:
            genres = df['label'].unique().tolist()
            return sorted(genres)
        else:
            print("Warning: 'label' column not found in DataFrame.")
            return []

    def create_sets(self):
        if self.own_long_features is None:
            print("Error: Feature dataset is not loaded. Please call get_feature_dataset() first.")
            return [], [], []
        
        labels = self.own_long_features['label']
        features = self.own_long_features.select_dtypes(include=[np.number])

        files_by_genre = defaultdict(list)
        for filename, genre in labels.items():
            files_by_genre[genre].append(filename)

        train_ids = []
        val_ids = []
        test_ids = []
        
        # Fixed seed
        random.seed(42)

        # Split the files into train, validation, and test sets
        for genre, filenames in files_by_genre.items():
            random.shuffle(filenames)
            num_files = len(filenames)
            train_end = int(0.7 * num_files)
            val_end = train_end + int(0.15 * num_files)

            train_ids.extend(filenames[:train_end])
            val_ids.extend(filenames[train_end:val_end])
            test_ids.extend(filenames[val_end:])

        # Create DataFrames for each set, df being the features
        X_train_df = features.loc[train_ids]
        X_val_df = features.loc[val_ids]
        X_test_df = features.loc[test_ids]

        # Feature scaling for more effective training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_df)
        X_val_scaled = scaler.transform(X_val_df)
        X_test_scaled = scaler.transform(X_test_df)
        
        X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_df.index, columns=X_train_df.columns)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, index=X_val_df.index, columns=X_val_df.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test_df.index, columns=X_test_df.columns)

        # Data augmentation
        noise_level = 0.1
        noise = np.random.randn(*X_train_scaled_df.shape) * noise_level
        X_train_augmented_df = X_train_scaled_df + noise

        # Assembles the triplet dataset
        def assemble_set(scaled_features_df):
            data_set = []
            for filename, feature_vector in scaled_features_df.iterrows():
                path = filename
                genre_string = labels.loc[filename]
                data_set.append((path, feature_vector.values, genre_string))
            return data_set

        train_set = assemble_set(X_train_augmented_df)
        val_set = assemble_set(X_val_scaled_df)
        test_set = assemble_set(X_test_scaled_df)

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

        return train_set, val_set, test_set