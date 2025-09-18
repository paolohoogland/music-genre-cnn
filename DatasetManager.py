from fileinput import filename
import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from Audio import Audio

class DatasetManager:
    OWN_LONG_FEATURES_PATH = 'Data/own_features_30_sec.csv'
    SCALER_PATH = 'scaler.joblib'

    def __init__(self):
        self.own_long_features = None
        self.all_genres_list = []
        self.scaler = None

    def extract_audio_features(self, audio_path):
        audio_instance = Audio()
        all_features_list = []
        print(f"Starting audio feature extraction for directory: {audio_path}")
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
                        mfcc_means, mfcc_vars = audio_instance.extract_mfccs(y, sr)
                        for i in range(20):
                            features[f'mfcc{i+1}_mean'] = mfcc_means[i]
                            features[f'mfcc{i+1}_var'] = mfcc_vars[i]
                        features['label'] = os.path.basename(root)
                        all_features_list.append(features)
                        print(f"Successfully processed: {filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        return pd.DataFrame(all_features_list)
    
    def create_feature_dataset(self):
        if os.path.exists(self.OWN_LONG_FEATURES_PATH):
            os.remove(self.OWN_LONG_FEATURES_PATH)
        try: 
            features_df = self.extract_audio_features("./Data/genres_original")
            features_df.to_csv(self.OWN_LONG_FEATURES_PATH, index=False)
            self.all_genres_list = features_df['label'].unique().tolist()
        except Exception as e:
            print(f"Error processing audio features: {e}")
        if 'features_df' in locals():
            print(f"Extracted {len(self.all_genres_list)} unique genres from audio features.")
            print("\n--- Feature Extraction Complete ---")
            print("Head of the new DataFrame:")
            print(features_df.head())

    def get_feature_dataset(self):
        try:
            df = pd.read_csv(self.OWN_LONG_FEATURES_PATH)
            self.all_genres_list = self.get_genre_list(df)
            # Use .set_index() on the original df, not on the return value of get_genre_list
            self.own_long_features = df.set_index('filename')
        except FileNotFoundError:
            print(f"File not found: {self.OWN_LONG_FEATURES_PATH}")
        except Exception as e:
            print(f"Error processing audio features: {e}")
            
    def get_genre_list(self, df):
        if 'label' in df.columns:
            return sorted(df['label'].unique().tolist())
        else:
            return []

    def create_sets(self):
        if self.own_long_features is None:
            print("Error: Feature dataset is not loaded. Please call get_feature_dataset() first.")
            return [], [], []
        
        # Force tempo to numeric, same as before
        if 'tempo' in self.own_long_features.columns:
            self.own_long_features['tempo'] = pd.to_numeric(self.own_long_features['tempo'], errors='coerce')
            # Fix the FutureWarning
            median_tempo = self.own_long_features['tempo'].median()
            self.own_long_features['tempo'].fillna(median_tempo, inplace=True)

        labels = self.own_long_features['label']
        features = self.own_long_features.select_dtypes(include=[np.number])
        
        # (Splitting logic remains the same)
        files_by_genre = defaultdict(list)
        for filename, genre in labels.items():
            files_by_genre[genre].append(filename)
        train_ids, val_ids, test_ids = [], [], []
        random.seed(42)
        for genre, filenames in files_by_genre.items():
            random.shuffle(filenames)
            num_files = len(filenames)
            train_end = int(0.7 * num_files)
            val_end = train_end + int(0.15 * num_files)
            train_ids.extend(filenames[:train_end])
            val_ids.extend(filenames[train_end:val_end])
            test_ids.extend(filenames[val_end:])

        X_train_df = features.loc[train_ids]
        X_val_df = features.loc[val_ids]
        X_test_df = features.loc[test_ids]

        # --- NEW: ROBUSTLY REMOVE CONSTANT FEATURES FROM THE TRAINING SET ---
        # Calculate standard deviation ON THE TRAINING SPLIT
        train_std = X_train_df.std()
        # Get the names of columns that are NOT constant
        non_constant_columns = train_std[train_std > 0].index.tolist()
        
        if len(non_constant_columns) < len(X_train_df.columns):
            constant_columns = set(X_train_df.columns) - set(non_constant_columns)
            print("\n" + "="*50)
            print("WARNING: Found constant columns in the training split. Removing them:")
            for col in constant_columns:
                print(f"- {col}")
            print("="*50 + "\n")

        # Keep only the non-constant columns for all splits
        X_train_df = X_train_df[non_constant_columns]
        X_val_df = X_val_df[non_constant_columns]
        X_test_df = X_test_df[non_constant_columns]
        # ---
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_df)
        joblib.dump(scaler, self.SCALER_PATH)
        print(f"\nScaler fitted on training data and saved to {self.SCALER_PATH}")
        
        X_val_scaled = scaler.transform(X_val_df)
        X_test_scaled = scaler.transform(X_test_df)
        
        X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_df.index, columns=X_train_df.columns)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, index=X_val_df.index, columns=X_val_df.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test_df.index, columns=X_test_df.columns)

        noise_level = 0.1
        noise = np.random.randn(*X_train_scaled_df.shape) * noise_level
        X_train_augmented_df = X_train_scaled_df + noise

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

    def process_single_audio(self, audio_path):
        try:
            self.scaler = joblib.load(self.SCALER_PATH)
        except FileNotFoundError:
            print(f"Error: Scaler file not found at {self.SCALER_PATH}. Please train a model first.")
            return None

        # --- MODIFIED: Load the list of features the scaler was trained on ---
        # We get this directly from the scaler object itself
        scaler_features = self.scaler.get_feature_names_out()

        raw_features_df = self._extract_single_file_features(audio_path)
        if raw_features_df is None: return None

        # Ensure the new file's features match the scaler's features
        try:
            raw_features_df = raw_features_df[scaler_features]
        except KeyError:
            print("Error: The features extracted from the new audio file do not match the features used for training.")
            return None
        
        scaled_features = self.scaler.transform(raw_features_df)
        print("Features extracted and scaled successfully.")
        return scaled_features

    def _extract_single_file_features(self, file_path):
        if not file_path.endswith('.wav'):
            print("Error: Provided file is not a .wav file.")
            return None
        audio_instance = Audio()
        print(f"Starting feature extraction for single file: {file_path}")
        try: 
            y, sr = audio_instance.load_audio(file_path)
            features = {}
            features['chroma_stft_mean'], features['chroma_stft_var'] = audio_instance.extract_chroma_sftf(y, sr)
            features['rms_mean'], features['rms_var'] = audio_instance.extract_rms(y)
            features['spectral_centroid_mean'], features['spectral_centroid_var'] = audio_instance.extract_spectral_centroid(y, sr)
            features['spectral_bandwidth_mean'], features['spectral_bandwidth_var'] = audio_instance.extract_spectral_bandwith(y, sr)
            features['rolloff_mean'], features['rolloff_var'] = audio_instance.extract_rolloff(y, sr)
            features['zero_crossing_rate_mean'], features['zero_crossing_rate_var'] = audio_instance.extract_zero_crossing_rate(y)
            features['harmony_mean'], features['harmony_var'] = audio_instance.extract_harmony(y, sr)
            features['perceptr_mean'], features['perceptr_var'] = audio_instance.extract_perceptr(y, sr)
            features['tempo'] = audio_instance.extract_tempo(y, sr)
            mfcc_means, mfcc_vars = audio_instance.extract_mfccs(y, sr)
            for i in range(20):
                features[f'mfcc{i+1}_mean'] = mfcc_means[i]
                features[f'mfcc{i+1}_var'] = mfcc_vars[i]
            return pd.DataFrame([features])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None