import cv2
import numpy as np
from app.utils.image import resize_if_needed

# ========================
# PREPROCESADO OPCIONAL
# ========================

def detect_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

    return None

def crop_document(image):
    contour = detect_document(image)
    if contour is None:
        return image

    pts = contour.reshape(4, 2)
    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(denoised, -1, kernel)

    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

# ========================
# CORE
# ========================

def blend_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    good_matches = []
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 10:
        # fallback simple
        w = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, int(img1.shape[0]*w/img1.shape[1])))
        img2 = cv2.resize(img2, (w, int(img2.shape[0]*w/img2.shape[1])))
        return np.vstack((img1, img2))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)

    all_corners = np.concatenate((
        np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2),
        warped_corners
    ), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-x_min, -y_min]
    H_translation = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    output = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    canvas = np.zeros_like(output)
    canvas[translation[1]:h1+translation[1], translation[0]:w1+translation[0]] = img1

    mask1 = np.zeros(output.shape[:2], dtype=np.uint8)
    mask1[translation[1]:h1+translation[1], translation[0]:w1+translation[0]] = 255

    mask2 = cv2.warpPerspective(np.ones((h2,w2), dtype=np.uint8)*255, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    overlap = cv2.bitwise_and(mask1, mask2)

    if cv2.countNonZero(overlap) > 0:
        mean1 = cv2.mean(canvas, mask=overlap)[:3]
        mean2 = cv2.mean(output, mask=overlap)[:3]
        diff = np.array(mean1) - np.array(mean2)

        output = np.clip(output.astype(np.float32) + diff, 0, 255).astype(np.uint8)

    # feather blending
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
    alpha1 = np.clip(dist1 / 40.0, 0, 1)
    alpha1[mask2 == 0] = 1
    alpha2 = 1 - alpha1

    alpha1 = np.expand_dims(alpha1, 2)
    alpha2 = np.expand_dims(alpha2, 2)

    result = (canvas * alpha1 + output * alpha2).astype(np.uint8)

    return result


def process_images(img1, img2, enhance=False, crop=False):
    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)

    if crop:
        img1 = crop_document(img1)
        img2 = crop_document(img2)

    if enhance:
        img1 = enhance_image(img1)
        img2 = enhance_image(img2)

    return blend_images(img1, img2)