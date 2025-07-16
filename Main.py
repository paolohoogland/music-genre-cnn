from Audio import Audio
from DatasetManager import DatasetManager

def main():
    audio_instance = Audio()
    dataset_mgr_instance = DatasetManager()

    # try:
    #     y, sr = audio_instance.load_audio("example.wav")
    #     print("Audio loaded successfully.")
    # except ValueError as e:
    #     print(f"Error loading audio: {e}")
    #     return
    # except FileNotFoundError:
    #     print("Audio file not found.")
    #     return
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    #     return

    # try:
    #     audio_dir="Data/genres_original", 
    #     output_dir="Data/processed_imgs",
    #     image_size = (256, 128)

    #     dataset_mgr_instance.create_dataset(audio_dir=audio_dir, output_dir=output_dir, size=image_size)

    
if __name__ == "__main__":
    main()