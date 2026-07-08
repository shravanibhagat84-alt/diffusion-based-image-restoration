"""
FastAPI Image Restoration - Final Working Version
Using simple but effective image processing
"""
import io
import base64
import uuid
from pathlib import Path
import numpy as np

import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image


app = FastAPI(title="Image Restoration")

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")


def apply_deblur(image: np.ndarray) -> np.ndarray:
    """Simple but effective deblurring using unsharp mask"""
    # Convert to LAB for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply unsharp mask to L channel
    blurred = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 1.0)
    sharpened = cv2.addWeighted(lab[:, :, 0], 1.5, blurred, -0.5, 0)
    lab[:, :, 0] = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Convert back
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Additional detail enhancement
    detail = cv2.detailEnhance(result, sigma_s=10, sigma_r=0.15)
    result = cv2.addWeighted(result, 0.7, detail, 0.3, 0)
    
    return result


def apply_derain(image: np.ndarray) -> np.ndarray:
    """Deraining using bilateral filter and morphology"""
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]
    
    # Bilateral filter for edge-preserving smoothing
    filtered = cv2.bilateralFilter(y, 9, 75, 75)
    
    # Morphological operations to remove rain
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    opened = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    
    # Estimate and subtract rain
    rain_estimate = cv2.absdiff(filtered, opened)
    y_derained = cv2.subtract(filtered, rain_estimate)
    
    ycrcb[:, :, 0] = y_derained
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    # Final denoising
    result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
    
    return result


def enhance_image(image: np.ndarray, task: str) -> np.ndarray:
    """Apply task-specific processing"""
    if task == 'deblurring':
        result = apply_deblur(image)
    else:  # deraining
        result = apply_derain(image)
    
    # Final color/contrast enhancement
    # Auto CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Slight saturation boost
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 10)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return result


def pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', quality=95)
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Image Restoration</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 25px 70px rgba(0,0,0,0.5);
        }
        h1 { color: #302b63; text-align: center; margin-bottom: 5px; font-size: 2.2em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 20px; }
        .task-buttons { display: flex; justify-content: center; gap: 15px; margin: 20px 0; }
        .task-btn {
            padding: 12px 30px;
            border: 2px solid #302b63;
            background: white;
            color: #302b63;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .task-btn.active { background: #302b63; color: white; }
        .upload-area {
            border: 3px dashed #302b63;
            border-radius: 15px;
            padding: 50px;
            text-align: center;
            margin: 20px 0;
            cursor: pointer;
            background: #f8f9fa;
        }
        .upload-area input { display: none; }
        .preview-container { display: flex; gap: 20px; margin: 25px 0; flex-wrap: wrap; }
        .preview-box { flex: 1; min-width: 280px; text-align: center; }
        .preview-box h3 { color: #333; margin-bottom: 12px; }
        .preview-img { max-width: 100%; max-height: 350px; border-radius: 12px; display: none; }
        .process-btn {
            display: block;
            width: 100%;
            max-width: 320px;
            margin: 25px auto;
            padding: 16px;
            background: linear-gradient(135deg, #302b63 0%, #24243e 100%);
            color: white;
            border: none;
            border-radius: 35px;
            font-size: 18px;
            cursor: pointer;
        }
        .process-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .loading { text-align: center; display: none; padding: 20px; }
        .spinner { width: 45px; height: 45px; border: 4px solid #e9ecef; border-top: 4px solid #302b63; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 15px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .time { text-align: center; color: #22c55e; font-weight: 600; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ Image Restoration</h1>
        <p class="subtitle">Advanced Deblurring & Deraining</p>
        <div class="task-buttons">
            <button class="task-btn active" onclick="selectTask('deblurring')">🔍 Deblurring</button>
            <button class="task-btn" onclick="selectTask('deraining')">🌧️ Deraining</button>
        </div>
        <div class="upload-area" onclick="document.getElementById('file').click()">
            <div style="font-size: 60px;">📁</div>
            <p style="font-size: 18px;"><b>Click to upload image</b></p>
            <input type="file" id="file" accept="image/*" onchange="handleFile(this)">
        </div>
        <div class="preview-container">
            <div class="preview-box"><h3>📷 Original</h3><img id="original" class="preview-img"></div>
            <div class="preview-box"><h3>✨ Result</h3><img id="result" class="preview-img"></div>
        </div>
        <div class="loading" id="loading"><div class="spinner"></div><p>Processing...</p></div>
        <div class="time" id="time"></div>
        <button class="process-btn" id="processBtn" onclick="processImage()" disabled>✨ Process Image</button>
    </div>
    <script>
        let selectedFile = null, currentTask = 'deblurring';
        function selectTask(task) { currentTask = task; document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active')); event.target.classList.add('active'); }
        function handleFile(input) {
            selectedFile = input.files[0];
            if (selectedFile) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('original').src = e.target.result;
                    document.getElementById('original').style.display = 'inline-block';
                    document.getElementById('result').style.display = 'none';
                    document.getElementById('processBtn').disabled = false;
                };
                reader.readAsDataURL(selectedFile);
            }
        }
        async function processImage() {
            if (!selectedFile) return;
            const loading = document.getElementById('loading'), processBtn = document.getElementById('processBtn'), resultImg = document.getElementById('result'), timeDiv = document.getElementById('time');
            loading.style.display = 'block'; processBtn.disabled = true; resultImg.style.display = 'none';
            const startTime = performance.now();
            const formData = new FormData();
            formData.append('file', selectedFile); formData.append('task', currentTask);
            try {
                const response = await fetch('/process', { method: 'POST', body: formData });
                const data = await response.json();
                const endTime = performance.now();
                if (data.success) {
                    resultImg.src = 'data:image/png;base64,' + data.result;
                    resultImg.style.display = 'inline-block';
                    timeDiv.textContent = '⏱️ ' + ((endTime - startTime) / 1000).toFixed(2) + 's';
                } else { alert('Error: ' + data.detail); }
            } catch (e) { alert('Error: ' + e.message); }
            finally { loading.style.display = 'none'; processBtn.disabled = false; }
        }
    </script>
</body>
</html>"""


@app.post("/process")
async def process_image_endpoint(file: UploadFile = File(...), task: str = Form("deblurring")):
    try:
        request_id = str(uuid.uuid4())
        input_path = UPLOAD_DIR / f"{request_id}_input.png"
        output_path = RESULT_DIR / f"{request_id}_output.png"
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))
        
        print(f"Processing: {image.size}, task: {task}")
        image.save(input_path)
        
        result_np = enhance_image(image_np, task)
        
        result_pil = Image.fromarray(result_np)
        result_pil.save(output_path, quality=95)
        print(f"Saved: {output_path}")
        
        result_base64 = pil_to_base64(result_pil)
        
        return JSONResponse({
            "success": True,
            "result": result_base64,
            "message": f"Processed ({task})",
            "task": task
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": "cpu", "mode": "advanced-processing"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
