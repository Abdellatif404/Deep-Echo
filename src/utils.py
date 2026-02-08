import librosa
import numpy as np


class AudioPreprocessor:
	def __init__(self, sr=None, n_mels=128, db_min=-50, db_max=-10):
		self.sr = sr
		self.n_mels = n_mels
		self.db_min = db_min
		self.db_max = db_max

	def load_audio(self, file_path):
		"""Loads audio and converts to Mel-Spectrogram in dB."""
		waveform, n_sr = librosa.load(file_path, sr=self.sr)

		S = librosa.feature.melspectrogram(
			y=waveform,
			sr=n_sr,
			n_mels=self.n_mels
		)

		S_db = librosa.power_to_db(S, ref=1.0)
		return S_db
	
	def normalize_db(self, S_db):
		"""Min-Max normalize the dB values to [0, 1] range."""
		S_norm = np.clip(S_db, self.db_min, self.db_max)
		S_norm = (S_norm - self.db_min) / (self.db_max - self.db_min)
		return S_norm
	
	def transform(self, file_path):
		"""Full pipeline: Raw Audio to Normalized Spectrogram"""
		spec = self.load_audio(file_path)
		return self.normalize_db(spec)
	
