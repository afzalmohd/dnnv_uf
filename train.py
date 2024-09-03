import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_epochs = 40
saved_model_name = 'tf_model.h5'

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.995):
            print("\nGood accuracy so cancelling training!")
            self.model.stop_training = True

def get_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
    # tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()
    return model

def training(train_x, train_y, val_x, val_y, model):
    callbacks = myCallback()
    model.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=num_epochs, batch_size=128, validation_data=(val_x, val_y), callbacks=[callbacks])

    return model

def training_with_augmentation(train_x, train_y, val_x, val_y, model):
    callbacks = myCallback()
    # Define the data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,          # Randomly rotate images by 10 degrees
        width_shift_range=0.1,      # Randomly shift images horizontally by 10%
        height_shift_range=0.1,     # Randomly shift images vertically by 10%
        zoom_range=0.1              # Randomly zoom images by 10%
    )

    datagen.fit(x_train)

    model.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer=tf.optimizers.Adam(), metrics=['accuracy'])

    model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=num_epochs, validation_data=(val_x, val_y), callbacks=[callbacks])

    return model

def retrain_with_augmentation(train_x, train_y, val_x, val_y, model_path):
    model = tf.keras.models.load_model(model_path)
    callbacks = myCallback()
    # Define the data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,          # Randomly rotate images by 10 degrees
        width_shift_range=0.1,      # Randomly shift images horizontally by 10%
        height_shift_range=0.1,     # Randomly shift images vertically by 10%
        zoom_range=0.1              # Randomly zoom images by 10%
    )

    datagen.fit(train_x)

    model.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
    model.fit(datagen.flow(train_x, train_y, batch_size=128), epochs=num_epochs, validation_data=(val_x, val_y), callbacks=[callbacks])

    return model


model = get_model()

mnist = tf.keras.datasets.mnist

(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

x_train_full = x_train_full / 255.0
x_train_full = x_train_full.reshape(x_train_full.shape[0], 28, 28, 1)
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# trained_model = training(x_train, y_train, x_val, y_val, model)
trained_model = training_with_augmentation(x_train, y_train, x_val, y_val, model)
# trained_model = retrain_with_augmentation(x_train, y_train, x_val, y_val, saved_model_name)

print(f"Evaluation on test data")
test_loss, test_accuracy = trained_model.evaluate(x_test, y_test, verbose=2)

trained_model.save(saved_model_name)