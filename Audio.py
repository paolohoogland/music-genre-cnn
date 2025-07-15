import wave
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

class Audio:

    def load_audio(self, file_name):
        # Read-only mode
        audio = wave.open(file_name, 'rb')

        # Valid wav file if it's mono, not stereo
        if audio.getnchannels() != 1:
            raise ValueError("Audio file must be mono (1 channel)")

        return audio
    
    def compute_mel_spectrogram(self, audio, sample_rate):
        # Ensure the audio is in the correct format
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer")

        audio_data = audio.readframes(audio.getnframes())
        audio.close()

        # 16-bit PCM audio data for librosa lib
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        normalized_audio_data = audio_data / np.max(np.abs(audio_data))

        # Compute the Mel spectrogram
        mel_spectr = librosa.feature.melspectrogram(y=normalized_audio_data, sr=sample_rate, n_mels=128)
        # Convert to decibel units
        mel_spectr = librosa.power_to_db(mel_spectr, ref=np.max)

        return mel_spectr

    def _plot_mel_spectrogram(self, mel_spectrogram): # Plot the Mel spectrogram using librosa's display module
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram)
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel Spectrogram')
        plt.tight_layout()

    def plot_spectrogram(self, mel_spectrogram):
        self._plot_mel_spectrogram(mel_spectrogram)
        plt.show()

    def save_spectrogram(self, mel_spectrogram, filename):
        self._plot_mel_spectrogram(mel_spectrogram)
        plt.savefig(filename)
        plt.close()
        return filename
    
