# Quick Start Script for Plant Disease Detection System

Write-Host "Plant Disease Detection System - Quick Start" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "What would you like to do?" -ForegroundColor Cyan
Write-Host "1. Install dependencies" -ForegroundColor White
Write-Host "2. Run application (development)" -ForegroundColor White
Write-Host "3. Run with Docker" -ForegroundColor White
Write-Host "4. Train model" -ForegroundColor White
Write-Host "5. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-5)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        
        # Create virtual environment if it doesn't exist
        if (-not (Test-Path "venv")) {
            Write-Host "Creating virtual environment..." -ForegroundColor Yellow
            python -m venv venv
        }
        
        # Activate virtual environment
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        .\venv\Scripts\Activate.ps1
        
        # Install requirements
        Write-Host "Installing Python packages..." -ForegroundColor Yellow
        pip install -r requirements.txt
        
        Write-Host ""
        Write-Host "[SUCCESS] Installation complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Run: .\venv\Scripts\Activate.ps1" -ForegroundColor White
        Write-Host "2. Run: python app.py" -ForegroundColor White
    }
    
    "2" {
        Write-Host ""
        Write-Host "Starting application..." -ForegroundColor Yellow
        
        # Check if model files exist
        if (-not (Test-Path "plant_disease_model.h5")) {
            Write-Host "[ERROR] Model file not found: plant_disease_model.h5" -ForegroundColor Red
            Write-Host "Please train the model first or download it." -ForegroundColor Yellow
            exit 1
        }
        
        if (-not (Test-Path "plant_disease_classes.npy")) {
            Write-Host "[ERROR] Classes file not found: plant_disease_classes.npy" -ForegroundColor Red
            exit 1
        }
        
        # Activate virtual environment if it exists
        if (Test-Path "venv\Scripts\Activate.ps1") {
            .\venv\Scripts\Activate.ps1
        }
        
        Write-Host ""
        Write-Host "Starting server..." -ForegroundColor Green
        Write-Host "Web Interface: http://localhost:5000" -ForegroundColor Cyan
        Write-Host "API Documentation: See README.md" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
        Write-Host ""
        
        python app.py
    }
    
    "3" {
        Write-Host ""
        Write-Host "Starting with Docker..." -ForegroundColor Yellow
        
        # Check if Docker is installed
        try {
            $dockerVersion = docker --version 2>&1
            Write-Host "[OK] $dockerVersion" -ForegroundColor Green
        } catch {
            Write-Host "[ERROR] Docker not found. Please install Docker Desktop." -ForegroundColor Red
            exit 1
        }
        
        # Check if model files exist
        if (-not (Test-Path "plant_disease_model.h5")) {
            Write-Host "[ERROR] Model file not found: plant_disease_model.h5" -ForegroundColor Red
            exit 1
        }
        
        Write-Host ""
        Write-Host "Building and starting Docker containers..." -ForegroundColor Yellow
        docker-compose up -d
        
        Write-Host ""
        Write-Host "Application started!" -ForegroundColor Green
        Write-Host "Web Interface: http://localhost:5000" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Useful commands:" -ForegroundColor Yellow
        Write-Host "  View logs: docker-compose logs -f web" -ForegroundColor White
        Write-Host "  Stop: docker-compose down" -ForegroundColor White
        Write-Host "  Restart: docker-compose restart" -ForegroundColor White
    }
    
    "4" {
        Write-Host ""
        Write-Host "Training model..." -ForegroundColor Yellow
        Write-Host "[WARNING] This requires the dataset to be downloaded" -ForegroundColor Yellow
        Write-Host ""
        
        # Activate virtual environment if it exists
        if (Test-Path "venv\Scripts\Activate.ps1") {
            .\venv\Scripts\Activate.ps1
        }
        
        # Check if dataset exists
        if (-not (Test-Path "archive\Plant Diseases Dataset")) {
            Write-Host "[ERROR] Dataset not found in archive/" -ForegroundColor Red
            Write-Host "Please download and extract the dataset first." -ForegroundColor Yellow
            exit 1
        }
        
        Write-Host "Starting training... This may take a while." -ForegroundColor Yellow
        python train_disease_model.py
        
        Write-Host ""
        Write-Host "[SUCCESS] Training complete!" -ForegroundColor Green
    }
    
    "5" {
        Write-Host "Goodbye!" -ForegroundColor Green
        exit 0
    }
    
    default {
        Write-Host "Invalid choice. Please run the script again." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
