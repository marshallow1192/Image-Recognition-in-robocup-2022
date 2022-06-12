from tensorflow.keras.preprocessing.image import ImageDataGenerator
from  tensorflow.keras.applications import ResNet50, vgg16
from  tensorflow.keras.models import Sequential, Model
from  tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from  tensorflow.keras import optimizers
from tensorflow import keras

image_data_genelator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   rotation_range=10)
 
train_generator = image_data_genelator.flow_from_directory(
    "./dataset/mask-model",
    target_size = (128,128),
    color_mode='rgb',
    classes = ["model-H","model-S","model-U"],
    class_mode='categorical',
    batch_size = 4,
    shuffle=True
)

test_generator = image_data_genelator.flow_from_directory(
    "./dataset/test-mask-model",
    target_size = (128,128),
    color_mode='rgb',
    classes = ["model-H","model-S","model-U"],
    class_mode='categorical',
    batch_size = 4,
    shuffle=True
)
 
input_tensor = Input(shape=(128, 128, 3))
base_model = vgg16.VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
 
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()

checkpoint = keras.callbacks.ModelCheckpoint("newmodel.hdf5")
 
history = model.fit_generator(
    train_generator ,
    validation_data=test_generator ,
    epochs=10,
    verbose=1,
    callbacks=[early_stopping,checkpoint])