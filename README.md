# Latent-IRSDE: Efficient Image Restoration with Latent Diffusion

High-resolution aerial image restoration for deblurring and deraining tasks using deep learning.

## Features

- **Deblurring**: Remove motion blur from images
- **Deraining**: Remove rain streaks from images
- **High-Resolution Support**: Up to 1920×1080 pixels
- **Web UI**: Easy-to-use interface for image processing
- **REST API**: Programmatic access to image restoration

## Requirements

### Hardware
- GPU with CUDA support (recommended for faster processing)
- At least 4GB RAM
- 10GB free disk space

### Software
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- cuDNN 8.0+

## Installation

### 1. Clone the Repository
```
bash
cd c:/Users/Gaurav\ t\ Satpute/Downloads/sakshuuu
```

### 2. Create Virtual Environment (Recommended)
```
bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```
bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Using run_server.py (Recommended)
```
bash
python run_server.py
```

### Option 2: Direct Python Execution
```
bash
python app.py
```

### Option 3: Using uvicorn
```
bash
uvicorn app:app --host 0.0.0.0 --port 8000

Open browser: http://localhost:8000
```

## Accessing the Application

After starting the server, open your browser and navigate to:
- **Web UI**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs

## Using the Web Interface

1. Open http://localhost:8000 in your browser
2. Select task type (Deblurring or Deraining)
3. Click "Choose Image" to upload an image
4. Click "Process Image" to restore the image
5. Download the restored image

## API Usage

### Process Image via curl
```
bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@your_image.png" \
  -F "task=deblurring"
```

### Python Example
```
python
import requests

url = "http://localhost:8000/process"
files = {"file": open("image.png", "rb")}
data = {"task": "deblurring"}

response = requests.post(url, files=files, data=data)
result = response.json()
# result['result'] contains base64 encoded image
```

## Project Structure

```
.
├── app.py                 # Main FastAPI application
├── run_server.py          # Server startup script
├── train.py               # Training script
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── models/
│   ├── autoencoder.py     # Autoencoder for latent space
│   ├── degradation_net.py # Degradation prediction network
│   ├── diffusion_unet.py  # Diffusion UNet model
│   └── refinement_net.py  # Refinement network
├── dataset/
│   ├── train/             # Training images
│   └── test/              # Test images
├── uploads/               # Uploaded images
└── results/               # Processed results
```

## Training

To train the models:
```
bash
python train.py
```

## Troubleshooting

### Port Already in Use
If you get "port 8000 is already in use":
```
bash
# Find and kill the process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### CUDA Out of Memory
Reduce image size or adjust tile_size in app.py

### Model Weights Not Found
The models will initialize with random weights. Training is required for best results.

## Configuration

Key settings in `app.py`:
```
python
MAX_RESOLUTION = (1920, 1080)  # Max image resolution
MODEL_CONFIG = {
    'degradation_channels': 32,
    'diffusion_channels': 64,
    'refiner_channels': 64,
}
```

## License

This project is for research purposes.
