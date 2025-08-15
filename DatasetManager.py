import os
import random
from collections import defaultdict
import pandas as pd
import re

class DatasetManager:
    LONG_FEATURES_PATH = 'Data/features_30_sec.csv'
    SHORT_FEATURES_PATH = 'Data/features_3_sec.csv'

    def __init__(self):
        # This list will hold (image_path, numeric_features, genre_name)
        self.long_features_af = self._load_features(self.LONG_FEATURES_PATH)
        self.short_features_af = self._load_features(self.SHORT_FEATURES_PATH)

        # Normalize features
        epsilon = 1e-6

        self.long_features_af = (self.long_features_af - self.long_features_af.mean()) / (self.long_features_af.std() + epsilon)
        self.short_features_af = (self.short_features_af - self.short_features_af.mean()) / (self.short_features_af.std() + epsilon)

    def _load_features(self, path):
        # af = audio features, 30 seconds snippet
        af = pd.read_csv(path)

        af = af.drop(columns=['label'])
        af = af.set_index('filename')

        return af
    
    def get_all_data_files(self):
        all_data_with_features =[]

        # Regex patterns
        long_pattern = re.compile(r"([a-zA-Z]+)\.\d{5}\.wav$")
        short_pattern = re.compile(r"([a-zA-Z]+)\.\d{5}\.\d+\.wav$")

        # Process long (30 sec) files
        for fname, features in self.long_features_af.iterrows():
            match = long_pattern.match(fname)
            if match:
                genre = match.group(1)
                all_data_with_features.append((fname, features.values, genre))

        # Process short (3 sec) files
        for fname, features in self.short_features_af.iterrows():
            match = short_pattern.match(fname)
            if match:
                genre = match.group(1)
                all_data_with_features.append((fname, features.values, genre))

        return all_data_with_features
    
    def get_genre_list(self):
        genres = set(self.long_features_af.index.str.split('.').str[0])
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