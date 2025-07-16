from Audio import Audio
from DatasetManager import DatasetManager

def main():
    audio_instance = Audio()
    dataset_mgr_instance = DatasetManager()

    try:
        y, sr = audio_instance.load_audio("example.wav")
        print("Audio loaded successfully.")
    except ValueError as e:
        print(f"Error loading audio: {e}")
        return
    except FileNotFoundError:
        print("Audio file not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    try:
        saved_filename = audio_instance.save_spectrogram(y, sr, "mel_spectrogram.png")
        print(f"Mel spectrogram saved as {saved_filename}.")
    except Exception as e:
        print(f"Error saving mel spectrogram: {e}")

    try:
        img = dataset_mgr_instance.normalize_image("mel_spectrogram.png")
        print("Image normalized successfully.")
        dataset_mgr_instance.save_image(img, "normalized_image.png")
        print("Image saved successfully.")
    except FileNotFoundError:
        print("Image file not found.")

    
if __name__ == "__main__":
    main()