import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape=(128, 313, 1)):
	"""
	Builds a convolutional autoencoder for anomaly detection in spectrograms.
	"""
	inputs = layers.Input(shape=input_shape)

	# Encoder
	x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
	x = layers.MaxPooling2D((2, 2), padding='same')(x)
	x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D((2, 2), padding='same')(x)
	encoded = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

	# Decoder
	x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
	x = layers.UpSampling2D((2, 2))(x)
	x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = layers.UpSampling2D((2, 2))(x)
	x = layers.Cropping2D(cropping=((0, 0), (0, 3)))(x)
	decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

	autoencoder = models.Model(inputs, decoded)
	autoencoder.compile(optimizer='adam', loss='mse')

	return autoencoder
