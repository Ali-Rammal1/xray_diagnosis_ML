# X-Ray Diagnosis ML Web Application

A comprehensive web application for diagnosing chest X-ray images using deep learning. This system can classify X-ray images into Normal, Pneumonia, or Tuberculosis categories.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Data Processing](#data-processing)
- [Performance Metrics](#performance-metrics)
- [Technologies Used](#technologies-used)
- [Code Structure](#code-structure)
- [Docker Setup](#docker-setup)
- [Manual Installation](#manual-installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Datasets](#datasets)

## Project Overview

This application provides an intuitive web interface for medical professionals to upload and analyze chest X-ray images. The system leverages a deep learning model based on ResNet18 to classify the images into different diagnostic categories with high accuracy.

## Model Architecture

- **Base Architecture**: ResNet18
- **Modifications**:
  - Final fully connected layer replaced with a custom layer for 4-class classification
  - Dropout layer (p=0.5) added before final classification layer for regularization
- **Number of Parameters**: ~11.7 million
- **Output Classes**: 4 (Normal, Pneumonia, Tuberculosis, Unknown)
- **Framework**: PyTorch

## Data Processing

### Preprocessing Pipeline

1. **Image Conversion**: RGB to grayscale
2. **Window-Level Adjustment**: Contrast enhancement using window level (WL: 600, WW: 1500)
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Applied with clip limit of 2.0
4. **Resizing**: All images standardized to 512×512 pixels
5. **Normalization**: Mean (0.232080) and standard deviation (0.070931) normalization

### Dataset Split

- **Training**: 80% of data
- **Validation**: 10% of data
- **Testing**: 10% of data

### Data Augmentation

- Random horizontal flips
- Random rotations (±8°)
- Random brightness and contrast adjustments

## Performance Metrics

- **Overall Accuracy**: 99.65%
- **Class-wise Precision**:
  - Normal: 0.9884
  - Pneumonia: 0.9892
  - Tuberculosis: 0.9981
  - Unknown: 0.9992
- **Class-wise Recall**:
  - Normal: 0.9954
  - Pneumonia: 0.9883
  - Tuberculosis: 0.9978
  - Unknown: 0.9995
- **Weighted F1 Score**: 0.9965

## Technologies Used

### Frontend

- **Framework**: React.js
- **Styling**: Styled Components

### Backend

- **Server**: Flask (Python)
- **ML Framework**: PyTorch
- **Image Processing**: OpenCV
- **Cross-Origin Handling**: Flask-CORS

## Code Structure

The repository is organized into three main directories:

### Backend

Contains the Flask server implementation for handling API requests and ML model inference.

- **`server.py`**: Main server file that loads the ML model and provides endpoints for X-ray image analysis
- **`requirements.txt`**: Python dependencies for the backend

### Frontend

Contains all React components and assets for the user interface.

- **`src/`**: React source code
  - **`components/`**: UI components like image upload, results display, etc.
  - **`services/`**: API service connectors
  - **`styles/`**: Styled components and CSS
  - **`App.js`**: Main application component
  - **`index.js`**: Application entry point

### src (Model & Data Processing)

Contains the trained model and scripts used for data processing and model evaluation.

- **`best_model.pth`**: The trained PyTorch model file used for inference
- **`normalize_all.py`**: Script to apply normalization on dataset images and convert to .npy format
- **`data_loading.py`**: Code used for loading and preprocessing data during model training
- **`model_score.txt`**: File containing metrics and performance scores of the trained model
- **Utility Scripts**:
  - Scripts for checking duplicates in the dataset
  - Tools for inspecting and visualizing data
  - GPU utilization verification scripts

> **Note**: The actual dataset files are not included in the repository due to their large size. This means that scripts that directly access the dataset (like `normalize_all.py` and `data_loading.py`) will not work without the dataset files. To use the application, we recommend running the Docker container which has everything needed for the main functionality.

## Docker Setup

### Prerequisites

- Docker
- Docker Compose

### Running with Docker

1. Clone this repository:

   ```bash
   git clone https://github.com/Ali-Rammal1/xray_diagnosis_ML
   cd xray_diagnosis_ML
   ```

2. Build and start the application:

   ```bash
   docker-compose build
   docker-compose up
   ```

3. Access the application:

   - Frontend: http://localhost:3000

4. To stop the application:
   ```bash
   docker-compose down
   ```

## Manual Installation

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   python server.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Usage

1. Open the web application at http://localhost:3000
2. Upload a chest X-ray image using the interface
3. The system will process the image and display the diagnosis result
4. View detailed confidence scores for each category
5. Access additional features like AI second opinion and diagnostic summary

## Key Features

### Primary Diagnosis

- Fast and accurate classification of X-ray images
- Confidence scores for each potential diagnosis
- Visual heatmap highlighting regions of interest

### External AI Second Opinion

- Integration with Google's Gemini AI for a second diagnostic opinion
- Compare results between the primary model and Gemini's analysis
- Enhanced diagnostic confidence through multi-model validation

### Diagnostic Summary Generation

- Automatic generation of comprehensive diagnostic summaries
- Inclusion of key findings and potential considerations
- Exportable format for integration with electronic health records
- Customizable templates for different clinical contexts

## Datasets

The model was trained on a combination of the following publicly available datasets:

1. **Tuberculosis + Normal**:

   - [Tuberculosis TB Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
   - [Lungs Disease Dataset (4 Types)](https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types/data)

2. **Pneumonia + Normal**:
   - [Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
