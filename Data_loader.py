import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Project Requirements
IMG_SIZE = (224, 224) # Resizing images to 224x224
BATCH_SIZE = 64



def get_data_loaders(csv_path, images_dir):
    print("Loading CSV...")
    # Read the CSV. 
    # Note: The Kaggle Fashion dataset sometimes has a few corrupted text lines, so 'on_bad_lines' skips them safely.
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # The 'id' in the CSV is just a number (e.g., 15970). We need to append '.jpg' 
    # so the generator knows exactly what file to look for in your images folder.
    df['image_filename'] = df['id'].astype(str) + '.jpg'
    
    # Initialize the ImageDataGenerator with the required augmentations
    print("Initializing Data Augmentation...")
    datagen = ImageDataGenerator(
        rescale=1./255,              # Normalizes pixel values
        rotation_range=20,           # Rotation augmentation
        horizontal_flip=True,        # Random flip augmentation
        brightness_range=[0.8, 1.2], # Brightness adjustment
        validation_split=0.2         # Train-validation split
    )

    print("Building Data Generators...")
    # Training Data Generator
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=images_dir,
        x_col='image_filename',
        y_col='masterCategory',      # Uses categories like Apparel, Footwear, Accessories
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Validation Data Generator
    val_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=images_dir,
        x_col='image_filename',
        y_col='masterCategory',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

if __name__ == "__main__":
    # Use raw strings (r"...") for Windows paths to avoid slash errors
    csv_file_path = r"D:\Visual Product Search System\data\styles.csv"
    images_folder_path = r"D:\Visual Product Search System\data\images"
    
    train_gen, val_gen = get_data_loaders(csv_file_path, images_folder_path)
    
    if train_gen.samples > 0:
        print("\nSuccess! The data loader is working perfectly.")
        print(f"Categories found: {list(train_gen.class_indices.keys())}")
    else:
        print("\nError: No images were found. Double-check your folder paths.")