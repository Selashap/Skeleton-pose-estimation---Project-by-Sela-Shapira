import keras
import joblib
from keras import layers
from tensorflow import data as tf_data
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


image_size = (160, 120)
batch_size = 128

# Define a preprocessing function to scale down the images
def preprocess_image(image, label):
    image = preprocess_input(image)
    return image, label

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "pictures_for_data",
    validation_split=0.2,
    subset="both",
    seed=13,
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.map(lambda x, y: (x / 255, y))
val_ds = val_ds.map(lambda x, y: (x / 255, y))

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation="relu", input_shape=(160, 120, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(4, 4), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units = 34, activation="relu"))
model.add(Dense(units=4, activation="softmax"))

epochs = 8

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

joblib.dump(model, 'pose_recog_model.pkl')
