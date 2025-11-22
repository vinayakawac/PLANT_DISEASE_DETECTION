# Plant Disease Detection System

A simple web application that helps farmers and gardeners identify plant diseases from leaf images using artificial intelligence. Just upload a photo of a plant leaf, and the system will tell you if the plant is healthy or diseased, along with treatment recommendations.

## What Can This App Do?

- **Identify Plant Diseases**: Upload a leaf photo and get instant results
- **Treatment Advice**: Get practical recommendations on how to treat the disease
- **Batch Analysis**: Check multiple plant images at once (up to 10 images)
- **Track History**: View statistics of all your previous scans
- **Works Offline**: Once set up, works on your local computer without internet

## Supported Plants

The system can identify diseases in these plants:
- **Fruits**: Apple, Blueberry, Cherry, Grape, Orange, Peach, Raspberry, Strawberry
- **Vegetables**: Tomato, Potato, Pepper, Corn, Squash, Soybean

**Total: 39 different conditions across 14 plant types**

## How to Run This Application

### What You Need
- A computer with Windows, Mac, or Linux
- Python 3.10 or newer (download from [python.org](https://www.python.org/downloads/))
- About 10 minutes for setup

### Step-by-Step Setup (First Time Only)

**1. Download the Project**
   - Download and extract this folder to your computer
   - Open the folder in your file explorer

**2. Open Command Prompt / Terminal**
   - **Windows**: Press `Win + R`, type `cmd`, press Enter
   - **Mac/Linux**: Open Terminal from Applications

**3. Navigate to Project Folder**
   ```bash
   cd path/to/plant_disease_detection
   ```
   (Replace `path/to` with your actual folder location)

**4. Install Required Software**
   ```bash
   pip install -r requirements.txt
   ```
   This will take a few minutes. Wait for it to complete.

**5. Start the Application**
   ```bash
   python app.py
   ```
   
   You should see messages like:
   ```
   Model loaded successfully
   Running on http://127.0.0.1:5000
   ```

**6. Open in Browser**
   - Open your web browser (Chrome, Firefox, Edge, etc.)
   - Go to: `http://localhost:5000`
   - You'll see the Plant Disease Detection page!

### Using the Application

1. **Upload a Photo**
   - Click on the upload area or drag and drop a leaf image
   - Supported formats: JPG, JPEG, PNG (max 16MB)

2. **Analyze**
   - Click the "Analyze Image" button
   - Wait a few seconds for results

3. **View Results**
   - See the plant type and disease (if any)
   - Read treatment recommendations
   - View confidence score

4. **Batch Analysis** (Optional)
   - Click "Batch Upload" tab
   - Select multiple images (up to 10)
   - Analyze all at once

### Stopping the Application

- Go back to the command prompt/terminal
- Press `Ctrl + C` (or `Cmd + C` on Mac)
- The server will stop

## Troubleshooting Common Issues

**Problem: "Python not found"**
- Solution: Install Python from python.org and make sure to check "Add Python to PATH" during installation

**Problem: "Model file not found"**
- Solution: Make sure files `plant_disease_model.h5` and `plant_disease_classes.npy` are in the project folder

**Problem: "Address already in use"**
- Solution: Close the previous instance or restart your computer

**Problem: "Can't connect to localhost:5000"**
- Solution: Make sure the application is running (check the terminal for "Running on..." message)

**Problem: Low accuracy results**
- Solution: Use clear, well-lit photos of leaves with visible symptoms

## Tips for Best Results

1. **Take Clear Photos**
   - Use good lighting (natural daylight is best)
   - Focus on the leaf showing symptoms
   - Avoid blurry or dark images

2. **Image Guidelines**
   - Single leaf per image works best
   - Show the entire leaf if possible
   - Include visible disease symptoms

3. **Interpreting Results**
   - Confidence above 90% = Very reliable
   - Confidence 70-90% = Fairly reliable
   - Confidence below 70% = Consider retaking the photo

4. **When to Consult an Expert**
   - If results seem incorrect
   - For severe or spreading infections
   - Before applying any chemical treatments

## Project Files

Here's what each file does:

- **app.py** - Main application (runs the web server)
- **plant_disease_model.h5** - AI model that detects diseases
- **plant_disease_classes.npy** - List of diseases the model knows
- **treatment_recommendations.json** - Treatment advice for each disease
- **requirements.txt** - List of software dependencies
- **templates/** - Folder containing the web page design
- **uploads/** - Stores your uploaded images temporarily
- **results/** - Saves analysis results

## Technical Details (For Developers)

**Technology Stack:**
- Python 3.10+
- Flask (Web Framework)
- TensorFlow/Keras (AI Model)
- HTML/CSS/JavaScript (Frontend)

**Model Information:**
- Architecture: Convolutional Neural Network (CNN)
- Training Dataset: PlantVillage Dataset
- Accuracy: ~95% on validation data
- Input Size: 224x224 pixels
- Output: 38 disease classes

**API Endpoints:**
- `GET /` - Web interface
- `GET /health` - Check if server is running
- `POST /api/predict` - Analyze single image
- `POST /api/batch-predict` - Analyze multiple images
- `GET /api/stats` - View usage statistics

## Advanced: Running with Docker (Optional)

If you prefer using Docker:

```bash
docker-compose up -d
```

Access at: http://localhost:5000

Stop with:
```bash
docker-compose down
```

## Important Notes

- **Privacy**: All images are stored locally on your computer
- **Offline Use**: Once set up, no internet required
- **Storage**: Old images can be deleted from `uploads/` folder
- **Accuracy**: This is a tool to assist, not replace professional diagnosis
- **Dataset Folders**: The `archive/` and `PlantVillage/` folders contain training data (~1.7GB) and are excluded from GitHub. The pre-trained model (`plant_disease_model.h5`) is included, so you don't need these folders to run the app.

**These are the only classes and disease types that the model has been trained on. Please note that predictions are limited to the following categories:**
- **Pepper bell**: Bacterial spot, healthy
- **Potato**: Early blight, healthy, Late blight
- **Tomato**: Target Spot, Tomato mosaic virus, Tomato Yellow Leaf Curl Virus, Bacterial spot, Early blight, healthy, Late blight, Leaf Mold, Septoria leaf spot, Spider mites/Two-spotted spider mite.
Predictions for any category outside this list may not be accurate.

## Questions or Issues?

If something isn't working:
1. Check the "Troubleshooting" section above
2. Make sure all files are in the correct folder
3. Restart the application
4. Try restarting your computer

## About This Project

This application was created to help farmers and gardeners quickly identify plant diseases and take appropriate action. It uses artificial intelligence trained on thousands of plant images to provide accurate disease detection and practical treatment recommendations.


**Disclaimer**: This tool is meant to assist in plant disease identification. For serious or uncertain cases, always consult with agricultural experts or extension services.

