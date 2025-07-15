from Audio import Audio

def main():
    audio_instance = Audio()

    try:
        audio = audio_instance.load_audio("example.wav")
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
        mel_spectrogram = audio_instance.compute_mel_spectrogram(audio, audio.getframerate())
        print("Mel spectrogram computed successfully.")
        print(mel_spectrogram)
    except ValueError as e:
        print(f"Error computing mel spectrogram: {e}")

    # try:
    #     audio_instance.plot_spectrogram(mel_spectrogram)
    #     print("Mel spectrogram plotted successfully.")
    # except Exception as e:
    #     print(f"Error plotting mel spectrogram: {e}")

    try:
        saved_filename = audio_instance.save_spectrogram(mel_spectrogram, "mel_spectrogram.png")
        print(f"Mel spectrogram saved as {saved_filename}.")
    except Exception as e:
        print(f"Error saving mel spectrogram: {e}")
    
if __name__ == "__main__":
    main()