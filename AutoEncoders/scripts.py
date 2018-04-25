import numpy as np
from keras.datasets import mnist
from keras.layers import Dense,Input
from keras.models import Model

(X_train,y_train),(X_test,y_test) = mnist.load_data()

print(X_train.shape)
print(y_test.shape)

X_train = X_train.reshape(X_train.shape[0],784).astype('float32')
X_test = X_test.reshape(X_test.shape[0],784).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# input dimension (28x28 images flattened)
input_dimension = 784
# size of encoded representations (compression by 24.5 % as input is 784)
encoding_dimension = 32

input_image_vector   = Input(shape=(input_dimension,))
encoded_image_vector = Dense(encoding_dimension,activation='relu')(input_image_vector)
decoded_image_vector = Dense(input_dimension,activation='sigmoid')(encoded_image_vector)

autoencoder = Model(input_image_vector,decoded_image_vector)
encoder = Model(input_image_vector,encoded_image_vector)


encoded_input = Input(shape=(encoding_dimension,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input,decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train,X_train,epochs=100,batch_size=256,shuffle=True,validation_data=(X_test,X_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)


