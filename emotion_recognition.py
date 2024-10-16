import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
import cv2
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

train_data = ImageDataGenerator(
    horizontal_flip = True,
    shear_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    vertical_flip = True,
    rotation_range = 0.2,
    zoom_range = 0.2,
    rescale = 1./255
)

test_data = ImageDataGenerator(rescale = 1./255)

training_set = train_data.flow_from_directory(directory = 'images/train',
                                              batch_size = 64,
                                              color_mode = 'grayscale',
                                              target_size = (48, 48)
                                              )
test_set = test_data.flow_from_directory(directory = 'images/validation',
                                              batch_size = 64,
                                              color_mode = 'grayscale',
                                              target_size = (48, 48)
                                              )



def build_model():
    inputs = layers.Input(shape = (48, 48, 1))
    x = layers.Conv2D(128, kernel_size = 5, padding = 'same', activation = 'relu')(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, kernel_size = 4, padding = 'same', activation = 'relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256, activation = 'relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation = 'relu')(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(7, activation = 'softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer = tf.keras.optimizers.Adam(3e-4), loss = tf.losses.CategoricalCrossentropy(), metrics = ['accuracy'])

    return model

model = build_model()


model.fit(x = training_set, validation_data = test_set, epochs = 100, batch_size = 64)

model.save('saved_model')