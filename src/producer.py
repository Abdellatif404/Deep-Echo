import os
import time
import random
import glob
from utils import AudioPreprocessor

def run_producer(data_dir):
	"""
	Simulates a stream by picking a random .wav from ANY machine ID 
	and ANY status (normal/abnormal).
	"""
	processor = AudioPreprocessor()

	print('Stream Simulator Started...')
	print(f'Scanning directory: {data_dir}')
	search_pattern = os.path.join(data_dir, "id_*/*/*.wav")
	all_files = glob.glob(search_pattern)

	if not all_files:
		print(f"No .wav files found in {data_dir}. Please check the path.")
		return
	print(f'Total files available for streaming: {len(all_files)}')

	try:
		while True:
			file_path = random.choice(all_files)

			parts = file_path.split(os.sep)
			machine_id = parts[-3]
			status = parts[-2]
			file_name = parts[-1]

			spectrogram = processor.transform(file_path)
			print(f'[STREAM] Machine: {machine_id} | Status: {status.upper()} | File: {file_name}')
			print(f'		 Spectrogram shape: {spectrogram.shape}')
			time.sleep(5)
	except KeyboardInterrupt:
		print('Stream Simulator Stopped.')


if __name__ == "__main__":
	NORMAL_PUMP_PATH = "../data/raw/pump/"
	run_producer(NORMAL_PUMP_PATH)
