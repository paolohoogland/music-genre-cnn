import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

class Audio:

    def load_audio(self, file_name):
        # Read-only mode
        y, sr = librosa.load(file_name, sr=None, mono=True)

        # Valid file if it's mono, not stereo
        if y.ndim != 1:
            raise ValueError("Audio file must be mono (single channel)")
        
        # Ensure the audio is in the correct format
        if sr <= 0:
            raise ValueError("Sample rate must be a positive integer")
        
        return y, sr
    
    def compute_mel_spectrogram(self, y, sr):
        # Parameters for the Mel spectrogram
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        # Compute the Mel spectrogram
        mel_spectr = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Convert to decibel units
        mel_spectr_db = librosa.power_to_db(mel_spectr, ref=np.max, top_db=80)

        return mel_spectr_db

    def _plot_mel_spectrogram(self, mel_spectrogram, sr): # Plot the Mel spectrogram using librosa's display module
        fig = plt.figure(figsize=(12, 4))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(mel_spectrogram, sr=sr, cmap='magma', ax=ax)
        return fig

    def plot_spectrogram(self, y, sr):
        mel_spectrogram = self.compute_mel_spectrogram(y, sr)
        self._plot_mel_spectrogram(mel_spectrogram, sr)
        plt.show()

    def save_spectrogram(self, y, sr, filename):
        mel_spectrogram = self.compute_mel_spectrogram(y, sr)
        fig = self._plot_mel_spectrogram(mel_spectrogram, sr)        
        fig.savefig(filename, dpi=100, pad_inches=0)
        plt.close(fig)
        return filename