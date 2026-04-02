from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io

from app.services.stitching import blend_images, enhance_image, crop_document
from app.utils.image import read_image, resize_if_needed

router = APIRouter()

@router.post("/merge-images")
async def merge_images(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    enhance: bool = Form(False),
    crop: bool = Form(False)
):
    img1 = read_image(file1)
    img2 = read_image(file2)

    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)

    if crop:
        img1 = crop_document(img1)
        img2 = crop_document(img2)

    if enhance:
        img1 = enhance_image(img1)
        img2 = enhance_image(img2)

    result = blend_images(img1, img2)

    _, buffer = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")