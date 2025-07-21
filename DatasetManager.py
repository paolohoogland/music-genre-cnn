import os
import random

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

        random.seed(42) # For reproducibility (use of seed)
        random.shuffle(all_data)

        num_files = len(all_data)
        train_size = int(num_files * 0.7) # 70% for training
        val_size = int(num_files * 0.15) # 15% for validation
        # Remaining 15% for testing

        train_set = all_data[:train_size]
        val_set = all_data[train_size:train_size + val_size]
        test_set = all_data[train_size + val_size:]

        print(f"Total spectrograms: {num_files}")
        print(f"Training set size: {len(train_set)} images")
        print(f"Validation set size: {len(val_set)} images")
        print(f"Test set size: {len(test_set)} images")

        return train_set, val_set, test_set