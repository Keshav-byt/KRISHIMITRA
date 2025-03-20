import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import ResNet101V2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model   # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore

# Constants
DATA_DIR = "data/Pest/Processed_Images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  # Increased batch size
EPOCHS = 30  # Increased number of epochs

# Data Generators
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=40,  # Increased rotation range
    zoom_range=0.3,  # Increased zoom range
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load Pre-trained Model
base_model = ResNet101V2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False   # Freeze base model

# Add Custom Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)   # Increased number of units
x = Dropout(0.4)(x)  # Increased dropout rate
output = Dense(train_gen.num_classes, activation='softmax')(x)

# Create Model
model = Model(inputs=base_model.input, outputs=output)

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Adjusted learning rate
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("Models/Pest_Detection/pest_detection_model.h5", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6) # Adjusted factor and patience
]

# Train Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save Model
model.save("Models/Pest_Detection/pest_detection_model.h5")

print("Model training completed.")