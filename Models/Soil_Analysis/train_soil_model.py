import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
import numpy as np

# Load preprocessed data
X_train = pd.read_csv("Data/Soil/X_train_scaled.csv").values
X_test = pd.read_csv("Data/Soil/X_test_scaled.csv").values
y_train = pd.read_csv("Data/Soil/y_train.csv").values.ravel()
y_test = pd.read_csv("Data/Soil/y_test.csv").values.ravel()

# Calculate class weights to handle imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Define a more complex neural network
def create_model(input_shape):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")  # Binary classification
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])
    return model

# Use cross-validation for more robust training
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}/{n_splits}")
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    model = create_model(X_train.shape[1])
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=100,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[early_stopping]
    )
    
    val_accuracy = model.evaluate(X_val_fold, y_val_fold)[1]
    cv_scores.append(val_accuracy)
    models.append(model)
    
    print(f"Fold {fold + 1} validation accuracy: {val_accuracy:.4f}")

# Select the best model
best_model_idx = np.argmax(cv_scores)
final_model = models[best_model_idx]
print(f"Best model from fold {best_model_idx + 1} with validation accuracy: {cv_scores[best_model_idx]:.4f}")

# Evaluate on test set
test_loss, test_accuracy = final_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Get predictions with probability for test set
test_predictions = final_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, (test_predictions > 0.5).astype(int)))

# Save the trained model
final_model.save("Models/Soil_Analysis/soil_tabular_model.h5")