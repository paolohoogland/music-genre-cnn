from PIL import Image
from Audio import Audio
import os

class DatasetManager:
    def resize_image(self, image, size=(256, 128)):
        return image.resize(size)

    def save_image(self, image, output_path):
        image.save(output_path)

    # def create_dataset(self, audio_dir, output_dir, size=(256, 128)):
    #     audio_processor = Audio()

    #     # Ensure the output directory exists
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #         print(f"Created directory: {output_dir}")

    #     try:
-    # 1. Create list of audio files in the directories
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"Audio directory '{audio_dir}' does not exist.")

    # 1. Load the audio file
    # 2. Generate the spectrogram as a PIL Image in memory
    # 3. Resize (normalize) the image



        
