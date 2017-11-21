import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

#%%
# load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)
plt.imshow(X_train[0], cmap='gray')
plt.title('Class '+ str(y_train[0]))
plt.show()

# reshape the input to take the shape (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# rescale so that image pixel lies in the interval [0,1] instead of [0,255]
X_train/=255
X_test/=255

# one-hot encode the labels
number_of_classes = 10
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

#%%
# Use a simple model - a linear stack of layers
model = Sequential()

# Add a convolutional layer with 32 filters of size=(6,6)
model.add(Conv2D(32, (6, 6), activation='relu', input_shape=(28,28,1)))

# Check shape of output
print(model.output_shape)

# Normalize the matrix after a convolution layer so the scale of each dimension
# remains the same (it reduces the training time significantly)
BatchNormalization(axis=-1)

# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Add a dropout layer
model.add(Dropout(0.2))

# Add a second convolutional layer with 16 filters of size=(3,3)
model.add(Conv2D(16, (3, 3), activation='relu'))
BatchNormalization(axis=-1)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#%%
# Flatten convolutional layers before passing them as input to the fully 
# connected dense layers
model.add(Flatten())

# Add a fully connected layer with 128 neurons
model.add(Dense(128, activation='relu'))
BatchNormalization(axis=-1)
model.add(Dropout(0.2))

# Add a second fully connected layer with 64 neurons
model.add(Dense(64, activation='relu'))
BatchNormalization(axis=-1)
model.add(Dropout(0.2))

# Add an output layer with 10 neurons, one for each class. Use a softmax 
# activation function to output probability-like predictions for each class
model.add(Dense(number_of_classes, activation='softmax'))

# Print model summary
model.summary()

#%%
# Compile model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Apply Data Augmentation to training and test sets
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
test_gen = ImageDataGenerator()

# Generate batches of augmented data
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

# Train the model
model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5, 
                    validation_data=test_generator, validation_steps=10000//64)

# Evaluate the model
score = model.evaluate(X_test, Y_test)
print('\nTest accuracy: ', score[1])

# Predict classes of test data
predictions = model.predict_classes(X_test)

# Number of mislabelled images
print('\nNumber of mislabelled images: ',np.count_nonzero(y_test-predictions),
      '\nTotal number of images      : ',len(y_test))

#sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
#sub.to_csv('./output_cnn.csv', index=False)

