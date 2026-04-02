import numpy as np
import cv2

def read_image(file):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(image):
    _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()

def resize_if_needed(img, max_dim=2000):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img