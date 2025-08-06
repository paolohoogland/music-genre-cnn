import os
import random
from collections import defaultdict
import pandas as pd
import re

class DatasetManager:
    IMAGES_PATH = 'Data/images_original'
    FEATURES_PATH = 'Data/features_30_sec.csv'

    def __init__(self):
        # This list will hold (image_path, numeric_features, genre_name)
        self.features_af = self._load_features()

        # Normalize features
        self.features_af = (self.features_af - self.features_af.mean()) / self.features_af.std()  

    def _load_features(self): 
        # af = audio features, 30 seconds snippet
        af = pd.read_csv(self.FEATURES_PATH)

        # Drop unnecessary columns
        af = af.drop(columns=['length', 'label']) 
        af = af.set_index('filename')
        return af
    
    def get_all_data_files(self):
        all_data_with_features =[]

        # Get all music genres in the dataset
        genre_dirs = [d for d in os.listdir(self.IMAGES_PATH) if os.path.isdir(os.path.join(self.IMAGES_PATH, d))]

        # Regex to match the filename pattern
        # Example: 'pop.00056.wav'
        filename_pattern = re.compile(r"([a-zA-Z]+)(\d{5})")

        for genre in genre_dirs:
            genre_path = os.path.join(self.IMAGES_PATH, genre)
            image_files = os.listdir(genre_path)

            # Create a CSV filename for each image file
            # This is because there's less image files than numeric features
            for image_file in image_files:
                if image_file.lower().endswith(('.png')):
                    full_image_path = os.path.join(genre_path, image_file)
                    base_name = os.path.splitext(image_file)[0]
                    match = filename_pattern.match(base_name)

                    if match:
                        genre_part = match.group(1)  # 'pop'
                        number_part = match.group(2) # '00056'
                        
                        # This creates 'pop.00056.wav'
                        csv_filename = f"{genre_part}.{number_part}.wav"

                    if csv_filename in self.features_af.index:
                        numeric_features = self.features_af.loc[csv_filename].values
                        all_data_with_features.append((full_image_path, numeric_features, genre)) 

        return all_data_with_features
    
    def get_genre_list(self):
        genre_dirs = [d for d in os.listdir(self.IMAGES_PATH) 
                      if os.path.isdir(os.path.join(self.IMAGES_PATH, d)) and not d.startswith('.')]
        
        return sorted(genre_dirs)

    def create_sets(self):
        all_data = self.get_all_data_files()

        # Group files by genre
        files_by_genre = defaultdict(list)
        for path, features, genre in all_data:
            files_by_genre[genre].append((path, features, genre))

        train_set = []
        val_set = []
        test_set = []
        
        # Fixed seed
        random.seed(42)

        for genre, files in files_by_genre.items():
            random.shuffle(files)
            
            num_files_in_genre = len(files)
            train_end = int(0.7 * num_files_in_genre)
            val_end = train_end + int(0.15 * num_files_in_genre)

            train_set.extend(files[:train_end])
            val_set.extend(files[train_end:val_end])
            test_set.extend(files[val_end:])

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        
        print(f"Total spectrograms: {len(all_data)}")
        print(f"Training set size: {len(train_set)} images")
        print(f"Validation set size: {len(val_set)} images")
        print(f"Test set size: {len(test_set)} images")

        return train_set, val_set, test_set