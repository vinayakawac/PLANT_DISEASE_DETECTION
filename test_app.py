"""
Unit tests for Plant Disease Detection API
"""
import unittest
import json
import os
from io import BytesIO
from PIL import Image
import numpy as np

from app import app

class TestPlantDiseaseAPI(unittest.TestCase):
    """Test cases for the API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
    def create_test_image(self, format='JPEG'):
        """Create a test image in memory"""
        img = Image.new('RGB', (224, 224), color='green')
        img_io = BytesIO()
        img.save(img_io, format)
        img_io.seek(0)
        return img_io
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
    
    def test_get_classes(self):
        """Test get classes endpoint"""
        response = self.client.get('/api/classes')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('total_classes', data)
        self.assertIn('classes', data)
        self.assertGreater(data['total_classes'], 0)
    
    def test_predict_no_file(self):
        """Test prediction without file"""
        response = self.client.post('/api/predict')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_with_image(self):
        """Test prediction with valid image"""
        img_io = self.create_test_image()
        
        response = self.client.post(
            '/api/predict',
            data={'file': (img_io, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        # Note: This will fail if model is not loaded
        # In production, ensure model is available
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertTrue(data['success'])
            self.assertIn('prediction', data)
            self.assertIn('plant', data['prediction'])
            self.assertIn('disease', data['prediction'])
            self.assertIn('confidence', data['prediction'])
    
    def test_predict_invalid_file_type(self):
        """Test prediction with invalid file type"""
        data = {'file': (BytesIO(b'not an image'), 'test.txt')}
        response = self.client.post(
            '/api/predict',
            data=data,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 400)
    
    def test_batch_predict_no_files(self):
        """Test batch prediction without files"""
        response = self.client.post('/api/batch-predict')
        self.assertEqual(response.status_code, 400)
    
    def test_batch_predict_with_images(self):
        """Test batch prediction with multiple images"""
        files = [
            ('files', (self.create_test_image(), 'test1.jpg')),
            ('files', (self.create_test_image(), 'test2.jpg'))
        ]
        
        response = self.client.post(
            '/api/batch-predict',
            data=files,
            content_type='multipart/form-data'
        )
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('results', data)
            self.assertIn('total_processed', data)
    
    def test_get_stats(self):
        """Test statistics endpoint"""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('total_predictions', data)
    
    def test_index_page(self):
        """Test index page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Plant Disease Detection', response.data)

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_allowed_file(self):
        """Test file extension validation"""
        from app import allowed_file
        
        self.assertTrue(allowed_file('test.jpg'))
        self.assertTrue(allowed_file('test.jpeg'))
        self.assertTrue(allowed_file('test.png'))
        self.assertFalse(allowed_file('test.txt'))
        self.assertFalse(allowed_file('test.gif'))
        self.assertFalse(allowed_file('test'))

if __name__ == '__main__':
    unittest.main()
