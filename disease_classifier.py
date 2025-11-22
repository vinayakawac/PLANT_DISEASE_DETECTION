import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseClassifier:
    def __init__(self, img_height=224, img_width=224, batch_size=32, use_transfer_learning=True):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        self.class_names = None
        self.history = None

    def prepare_data(self, dataset_path):
        train_dir = os.path.join(dataset_path, "train")
        valid_dir = os.path.join(dataset_path, "valid")

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )

        valid_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.class_names = list(train_generator.class_indices.keys())
        print(f"Found {len(self.class_names)} classes:\n{self.class_names}")
        return train_generator, valid_generator

    def build_model(self):
        """Build model with optional transfer learning"""
        if self.use_transfer_learning:
            logger.info("Building model with MobileNetV2 transfer learning")
            # Use pre-trained MobileNetV2
            base_model = MobileNetV2(
                input_shape=(self.img_height, self.img_width, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False  # Freeze base model
            
            self.model = Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                BatchNormalization(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(len(self.class_names), activation='softmax')
            ])
        else:
            logger.info("Building custom CNN model")
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                
                Conv2D(256, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                
                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(len(self.class_names), activation='softmax')
            ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )
        self.model.summary()

    def train(self, train_gen, val_gen, epochs=15, checkpoint_path='best_model.h5'):
        """Train the model with callbacks"""
        logger.info(f"Starting training for {epochs} epochs")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history

    def save_model(self, model_path="disease_model.h5", class_path="disease_classes.npy"):
        """Save model and classes"""
        self.model.save(model_path)
        np.save(class_path, self.class_names)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Classes saved to {class_path}")
        
        # Save training history if available
        if self.history:
            import json
            history_path = model_path.replace('.h5', '_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f)
            logger.info(f"Training history saved to {history_path}")

    def load_model(self, model_path="disease_model.h5", class_path="disease_classes.npy"):
        """Load model and classes"""
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = np.load(class_path, allow_pickle=True)
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Classes loaded from {class_path}")
    
    def evaluate(self, test_gen):
        """Evaluate model on test data"""
        logger.info("Evaluating model...")
        results = self.model.evaluate(test_gen, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        logger.info("Evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics

    def predict(self, image_path):
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        idx = np.argmax(prediction)
        return {
            "class": self.class_names[idx],
            "confidence": f"{np.max(prediction) * 100:.2f}%"
        }
