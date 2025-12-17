
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import numpy as np

# Config
PROJECT_ROOT = Path(__file__).parent.parent
ORGANIZED_DATA_DIR = PROJECT_ROOT / "organized_data"
MODEL_PATH = Path(__file__).parent / "skin_model.h5"
IMG_SIZE = 224
BATCH_SIZE = 32

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    if not ORGANIZED_DATA_DIR.exists():
        print(f"Error: Data not found at {ORGANIZED_DATA_DIR}")
        return

    print("Loading model...")
    model = load_model(str(MODEL_PATH))

    print("Setting up data generator...")
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    validation_generator = val_datagen.flow_from_directory(
        ORGANIZED_DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print("\nEvaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)

    print("\n" + "="*30)
    print(f"Model Accuracy: {val_accuracy*100:.2f}%")
    print(f"Model Loss:     {val_loss:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
