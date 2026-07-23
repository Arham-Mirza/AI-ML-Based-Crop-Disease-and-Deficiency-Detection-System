import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
import json

def create_model(num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def prepare_dataset_splits():
    dataset_path = 'fruits_dataset'
    
    class_folders = [f for f in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, f))]
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    temp_train_dir = 'temp_train'
    temp_val_dir = 'temp_validation'
    
    for dir_path in [temp_train_dir, temp_val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        
        for class_folder in class_folders:
            os.makedirs(os.path.join(dir_path, class_folder))
    
    # Split each class into train/validation
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Class {class_folder}: {len(images)} images")
        
        if len(images) == 0:
            continue
            
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(temp_train_dir, class_folder, img)
            shutil.copy2(src, dst)
            
        for img in val_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(temp_val_dir, class_folder, img)
            shutil.copy2(src, dst)
    
    return temp_train_dir, temp_val_dir, class_folders

def train_model():
    print("Preparing dataset...")
    train_dir, val_dir, class_names = prepare_dataset_splits()
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    batch_size = 32
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(class_names)
    print(f"Training on {num_classes} classes: {class_names}")
    
    model = create_model(num_classes)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('fruithealth_model.h5')
    
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)
    
    train_loss, train_acc = model.evaluate(train_generator, verbose=0)
    val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
    
    print(f"\nTraining completed!")
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Model saved as 'fruithealth_model.h5'")
    print(f"Class names saved as 'class_names.pkl' and 'class_names.json'")

def predict_single_image(image_path):
    try:
        model = keras.models.load_model('fruithealth_model.h5')
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        print(f"\nPrediction for {image_path}:")
        print(f"Class: {class_names[predicted_idx]}")
        print(f"Confidence: {confidence:.4f}")
        
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        print("\nTop 3 predictions:")
        for idx in top_3_idx:
            print(f"  {class_names[idx]}: {predictions[0][idx]:.4f}")
            
        return class_names[predicted_idx], confidence
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

if __name__ == "__main__":
    # Check if model exists, if not train it
    if not os.path.exists('fruithealth_model.h5'):
        print("No trained model found. Starting training...")
        train_model()
    else:
        print("Trained model found. You can use predict_single_image() to test.")
    
    # Test prediction on example.jpg if it exists
    if os.path.exists('example.jpg'):
        print("\nTesting prediction on example.jpg...")
        predict_single_image('example.jpg')