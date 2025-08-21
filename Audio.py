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
    
    def extract_rms(self, y):
        rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0] # Convert to a 1D array
        return np.mean(rms_frames), np.var(rms_frames) # Return mean and variance of RMS energy
    
    def extract_spectral_centroid(self, y, sr):
        spectral_centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
        return np.mean(spectral_centroid_frames), np.var(spectral_centroid_frames)
    
    def extract_chroma_sftf(self, y, sr):
        chroma_rep = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        return np.mean(chroma_rep), np.var(chroma_rep)

    def extract_spectral_bandwith(self, y, sr):
        spectral_bandwidth_frames = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
        return np.mean(spectral_bandwidth_frames), np.var(spectral_bandwidth_frames)
    
    def extract_rolloff(self, y, sr):
        spectral_rolloff_frames = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=512)[0] # roll_percent=0.85
        return np.mean(spectral_rolloff_frames), np.var(spectral_rolloff_frames)    
    
    def extract_zero_crossing_rate(self, y):
        zero_crossing_rate_frames = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
        return np.mean(zero_crossing_rate_frames), np.var(zero_crossing_rate_frames)
    
    def extract_harmony(self, y, sr): # differs from the GTZAN dataset (harmony mean)
        harmony_frames = librosa.effects.harmonic(y=y)
        return np.mean(harmony_frames), np.var(harmony_frames)
    
    def extract_perceptr(self, y, sr): # differs from the GTZAN dataset (perceptr mean)
        percussive_data = librosa.effects.percussive(y=y)
        return np.mean(percussive_data), np.var(percussive_data)
    
    def extract_tempo(self, y, sr):
        tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
        return tempo
    
    def extract_mfccs(self, y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
        return np.mean(mfccs, axis=1), np.var(mfccs, axis=1)