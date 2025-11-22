"""
Configuration module for Plant Disease Detection API
"""
import os
from pathlib import Path

class Config:
    """Base configuration"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # File upload settings
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    UPLOAD_FOLDER = Path('uploads')
    RESULTS_FOLDER = Path('results')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'plant_disease_model.h5')
    CLASSES_PATH = os.environ.get('CLASSES_PATH', 'plant_disease_classes.npy')
    TREATMENT_DATA_PATH = 'treatment_recommendations.json'
    
    # Image settings
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    
    # Batch processing
    MAX_BATCH_SIZE = 10
    
    # Logging
    LOG_FILE = 'app.log'
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
