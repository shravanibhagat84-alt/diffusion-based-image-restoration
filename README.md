#  Diffusion-Based Image Restoration

##  Project Overview

Diffusion-Based Image Restoration is an AI-powered image enhancement system developed to restore degraded images by reducing blur and removing rain effects while preserving important visual details. The project uses deep learning and diffusion-based techniques to improve image quality and provide cleaner, sharper outputs.

The application includes a web interface built with FastAPI, allowing users to upload degraded images and obtain restored results efficiently.

---

##  Features
- **Deblurring**: Remove motion blur from images
- **Deraining**: Remove rain streaks from images
- **High-Resolution Support**: Up to 1920×1080 pixels
- **Web UI**: Easy-to-use interface for image processing
- **REST API**: Programmatic access to image restoration

---

## Technologies Used

* Python
* FastAPI
* Uvicorn
* OpenCV
* NumPy
* PyTorch
* PIL (Pillow)
* HTML/CSS
* Diffusion-Based Deep Learning Models

---

## Project Structure

```
Diffusion-Based-Image-Restoration/
│
├── dataset/
├── models/
├── uploads/
├── results/
├── app_final_working.py
├── run_server.py
├── start_server.py
├── train.py
├── losses.py
├── utils.py
├── README.md
└── .gitignore
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/shravanibhagat84-alt/diffusion-based-image-restoration.git
```

### Open the project

```bash
cd diffusion-based-image-restoration
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate

---

## Run the Project

Start the server using:

```
python run_server.py
```

Open your browser and visit:

```
http://localhost:8002
```

---

##  Sample Workflow

1. Launch the application.
2. Upload a degraded or blurry image.
3. The model processes the image.
4. View and download the restored image.

---

##  Project Objectives

* Improve degraded image quality.
* Restore blurred and rainy images.
* Learn diffusion-based deep learning techniques.
* Build a deployable AI image restoration application.

---

## Future Enhancements

* Support multiple image restoration models.
* GPU acceleration for faster inference.
* Batch image processing.
* Image quality metrics (PSNR and SSIM).
* Cloud deployment.
* Mobile-friendly interface.

---

##  Author

**Shravani Bhagat**

MCA (AI & Data Science)

DY Patil International University

