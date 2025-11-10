from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
import uvicorn
import numpy as np
import nibabel as nib
import io
import torch
import zipfile
import tempfile
import os
import base64
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from monai.transforms import (
    Compose, EnsureTyped, Orientationd, Spacingd,
    SpatialCropd, NormalizeIntensityd
)
import torch
import torch.nn as nn
import torch.nn.functional as F

app = FastAPI()

# Custom exception handler for large file uploads
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large. Maximum size allowed is 100MB."}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 413:
        return JSONResponse(
            status_code=413,
            content={"detail": "File too large. Maximum size allowed is 100MB."}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Allow CORS for local dev frontend
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(0.3)  # %30 dropout

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class UnetModel(nn.Module):
    def __init__(self, num_classes):
        super(UnetModel, self).__init__()

        self.encoder1 = UNetEncoder(4, 32)
        self.encoder2 = UNetEncoder(32, 64)
        self.encoder3 = UNetEncoder(64, 128)
        self.encoder4 = UNetEncoder(128, 256)

        self.bottleneck = UNetBottleneck(256, 512)

        self.decoder1 = UNetDecoder(512, 256)
        self.decoder2 = UNetDecoder(256, 128)
        self.decoder3 = UNetDecoder(128, 64)
        self.decoder4 = UNetDecoder(64, 32)

        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = self.final_conv(x)
        return x
    
# Define model architecture
model = UnetModel(4).to(device)

# Load state dict
model_path = r"C:\Users\abirg\Videos\works\potato-disease-classification-main\api\tumor_model.pth"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Define MONAI preprocessing pipeline
test_transforms = Compose([
    EnsureTyped(keys=["vol"]),
    Orientationd(keys=["vol"], axcodes="RAS"),
    Spacingd(keys=["vol"], pixdim=(1.2, 1.2, 1.0), mode="bilinear"),
    SpatialCropd(keys=["vol"], roi_center=[100, 100, 79], roi_size=[160, 160, 128]),
    NormalizeIntensityd(keys=["vol"], nonzero=True, channel_wise=True),
])

def extract_nifti_from_zip(zip_content):
    files = {}

    with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
        for file_info in zip_ref.filelist:
            if file_info.filename.endswith(('.nii', '.nii.gz')):
                file_content = zip_ref.read(file_info.filename)
                filename_lower = file_info.filename.lower()

                if 'flair' in filename_lower:
                    files['flair'] = (file_content, file_info.filename)
                elif 't1ce' in filename_lower:
                    files['t1ce'] = (file_content, file_info.filename)
                elif 't2' in filename_lower and 't1ce' not in filename_lower:
                    files['t2'] = (file_content, file_info.filename)
                elif 't1' in filename_lower and 't1ce' not in filename_lower:
                    files['t1'] = (file_content, file_info.filename)

    return files

def preprocess_image(vol_data):
    data = {"vol": vol_data}
    data = test_transforms(data)
    vol_tensor = data["vol"].unsqueeze(0).to(device)  # (1, C, H, W, D)
    return vol_tensor

def create_segmentation_overlay(original_slice, segmentation_slice, alpha=0.5):
    """Create an overlay of segmentation on original image"""
    # Normalize original slice to 0-255
    if original_slice.max() > original_slice.min():
        original_normalized = ((original_slice - original_slice.min()) / 
                             (original_slice.max() - original_slice.min()) * 255).astype(np.uint8)
    else:
        original_normalized = np.zeros_like(original_slice, dtype=np.uint8)
    
    # Convert to RGB
    original_rgb = cv2.cvtColor(original_normalized, cv2.COLOR_GRAY2RGB)
    
    # Create colored segmentation mask
    colors = {
        0: [0, 0, 0],      # Background - black
        1: [255, 0, 0],    # Tumor class 1 - red
        2: [0, 255, 0],    # Tumor class 2 - green
        3: [0, 0, 255],    # Tumor class 3 - blue
    }
    
    colored_seg = np.zeros((*segmentation_slice.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        mask = segmentation_slice == class_id
        colored_seg[mask] = color
    
    # Create overlay
    # Ensure both images have the same size
    if original_rgb.shape != colored_seg.shape:
        colored_seg = cv2.resize(colored_seg, (original_rgb.shape[1], original_rgb.shape[0]))

    overlay = cv2.addWeighted(original_rgb, 1-alpha, colored_seg, alpha, 0)
    
    return overlay

def numpy_to_base64(image_array):
    """Convert numpy array to base64 string"""
    # Convert to PIL Image
    if len(image_array.shape) == 3:  # RGB image
        image = Image.fromarray(image_array.astype(np.uint8))
    else:  # Grayscale
        image = Image.fromarray(image_array.astype(np.uint8), mode='L')
    
    # Save to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def process_prediction_to_images(prediction, original_data, num_slices=5):
    """Convert prediction array to displayable images"""
    # Remove batch dimension if present
    if len(prediction.shape) == 4:
        prediction = prediction[0]  # Shape: (H, W, D)
    
    # Get middle slices for visualization
    depth = prediction.shape[2]
    slice_indices = np.linspace(depth//4, 3*depth//4, num_slices, dtype=int)
    
    images = []
    for i, slice_idx in enumerate(slice_indices):
        # Get prediction slice
        pred_slice = prediction[:, :, slice_idx]
        
        # Get corresponding original slice (use FLAIR as reference)
        if len(original_data.shape) == 4:  # (C, H, W, D)
            original_slice = original_data[0, :, :, slice_idx]  # Use FLAIR channel
        else:
            original_slice = original_data[:, :, slice_idx]
        
        # Create overlay
        overlay = create_segmentation_overlay(original_slice, pred_slice)
        
        # Convert to base64
        img_base64 = numpy_to_base64(overlay)
        
        images.append({
            "slice_index": int(slice_idx),
            "image": img_base64,
            "type": "overlay"
        })
        
        # Also create pure segmentation image
        seg_colored = np.zeros((*pred_slice.shape, 3), dtype=np.uint8)
        colors = {
            1: [255, 0, 0],    # Red
            2: [0, 255, 0],    # Green  
            3: [0, 0, 255],    # Blue
        }
        
        for class_id, color in colors.items():
            mask = pred_slice == class_id
            seg_colored[mask] = color
            
        seg_base64 = numpy_to_base64(seg_colored)
        
        images.append({
            "slice_index": int(slice_idx),
            "image": seg_base64,
            "type": "segmentation_only"
        })
    
    return images

@app.post("/predict-zip")
async def predict_from_zip(
    zip_file: UploadFile = File(...)
):
    """Handle zip file upload containing BraTS data"""
    try:
        # Check file extension
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only ZIP files are supported")
        
        # Read zip content
        zip_content = await zip_file.read()
        
        # Extract NIfTI files
        nifti_files = extract_nifti_from_zip(zip_content)
        
        # Check if we have all required files
        required_files = ['flair', 't1ce', 't2', 't1']
        missing_files = [f for f in required_files if f not in nifti_files]
        
        if missing_files:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required files: {missing_files}. Expected files with 'flair', 't1ce', 't2', 't1' in filename."
            )
        
        # Save temporary files and load images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_paths = {}
            for key in ['flair', 't1ce', 't2', 't1']:
                file_bytes, original_name = nifti_files[key]
                ext = ".nii.gz" if original_name.endswith(".nii.gz") else ".nii"
                temp_path = os.path.join(temp_dir, f"{key}{ext}")
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)
                temp_paths[key] = temp_path

            # Load using nibabel
            flair_img = nib.load(temp_paths['flair'])
            t1ce_img = nib.load(temp_paths['t1ce'])
            t2_img = nib.load(temp_paths['t2'])
            t1_img = nib.load(temp_paths['t1'])

            # Get data
            flair_data = flair_img.get_fdata()
            t1ce_data = t1ce_img.get_fdata()
            t2_data = t2_img.get_fdata()
            t1_data = t1_img.get_fdata()

            # Stack and preprocess
            input_data = np.stack([flair_data, t1ce_data, t2_data, t1_data], axis=-1)
            input_data = np.moveaxis(input_data, -1, 0)  # (C, H, W, D)
            
            input_tensor = preprocess_image(input_data)
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).cpu().numpy()
            
            # Process prediction to images
            images = process_prediction_to_images(prediction, input_data)
            
            # Calculate some statistics
            unique_classes, counts = np.unique(prediction, return_counts=True)
            class_stats = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
            
            return {
                "prediction_shape": prediction.shape,
                "message": "Inference complete from ZIP file",
                "files_processed": list(nifti_files.keys()),
                "images": images,
                "class_statistics": class_stats,
                "class_labels": {
                    0: "Background",
                    1: "Necrotic/Non-enhancing tumor", 
                    2: "Peritumoral edema/invaded tissue",
                    3: "GD-enhancing tumor"
                }
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZIP file: {str(e)}")

@app.post("/predict")
async def predict(
    flair_file: UploadFile = File(...),
    t1ce_file: UploadFile = File(...),
    t2_file: UploadFile = File(...),
    t1_file: UploadFile = File(...)
):
    try:
        # Load images
        flair_img = nib.load(io.BytesIO(await flair_file.read()))
        t1ce_img = nib.load(io.BytesIO(await t1ce_file.read()))
        t2_img = nib.load(io.BytesIO(await t2_file.read()))
        t1_img = nib.load(io.BytesIO(await t1_file.read()))

        # Get data
        flair_data = flair_img.get_fdata()
        t1ce_data = t1ce_img.get_fdata()
        t2_data = t2_img.get_fdata()
        t1_data = t1_img.get_fdata()

        # Stack and preprocess
        input_data = np.stack([flair_data, t1ce_data, t2_data, t1_data], axis=-1)
        input_data = np.moveaxis(input_data, -1, 0)  # (C, H, W, D)

        input_tensor = preprocess_image(input_data)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).cpu().numpy()

        # Process prediction to images
        images = process_prediction_to_images(prediction, input_data)
        
        # Calculate some statistics
        unique_classes, counts = np.unique(prediction, return_counts=True)
        class_stats = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}

        return {
            "prediction_shape": prediction.shape,
            "message": "Inference complete",
            "images": images,
            "class_statistics": class_stats,
            "class_labels": {
                0: "Background",
                1: "Necrotic/Non-enhancing tumor", 
                2: "Peritumoral edema/invaded tissue",
                3: "GD-enhancing tumor"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

if __name__ == "__main__":
    print("🚀 Backend is running on http://localhost:8000")
    # Run with increased timeout for large file uploads
    uvicorn.run(
        app, 
        host="localhost", 
        port=8000,
        timeout_keep_alive=30
    )