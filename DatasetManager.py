import os
import random
from collections import defaultdict

class DatasetManager:
    IMAGES_PATH = 'Data/images_original'

    def __init__(self):
    # This list will hold (image_path, genre_name)
        self.all_files_with_labels = []

    def get_all_image_files(self):
        self.all_files_with_labels = []

        # Get all music genres in the dataset
        genre_dirs = [d for d in os.listdir(self.IMAGES_PATH) if os.path.isdir(os.path.join(self.IMAGES_PATH, d))]

        for genre in genre_dirs:
            genre_path = os.path.join(self.IMAGES_PATH, genre)
            image_files = os.listdir(genre_path)

            for image_file in image_files:
                if image_file.lower().endswith(('.png')):
                    full_image_path = os.path.join(genre_path, image_file)
                    self.all_files_with_labels.append((full_image_path, genre))

        return self.all_files_with_labels
    
    def get_genre_list(self):
        genre_dirs = [d for d in os.listdir(self.IMAGES_PATH) 
                      if os.path.isdir(os.path.join(self.IMAGES_PATH, d)) and not d.startswith('.')]
        
        return sorted(genre_dirs)

    def create_sets(self):
        all_data = self.get_all_image_files()

        # Group files by genre
        files_by_genre = defaultdict(list)
        for path, genre in all_data:
            files_by_genre[genre].append((path, genre))

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