import cv2
import numpy as np

def process_ecg_image(image_path):
    """
    Robust ECG signal extraction using Contour Detection.
    Instead of column-wise scanning which picks up vertical noise, 
    this approach finds the longest continuous lines in the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image: " + image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # --- Step 1: Preprocessing ---
    # Normalize and blur
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu's thresholding to get signal as white (255) on black (0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up small noise/grid dots
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # --- Step 2: Contour Discovery ---
    # Find all continuous line segments
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(width).tolist()

    # --- Step 3: Select the Best Contour ---
    # The ECG signal is typically the contour with the largest bounding box width
    # and a reasonable length-to-width ratio.
    signal_mask = np.zeros_like(binary)
    
    # Sort contours by width (horizontal extent)
    def get_width(c):
        x, y, w, h = cv2.boundingRect(c)
        return w

    sorted_contours = sorted(contours, key=get_width, reverse=True)
    
    # Combine a few of the widest contours in case the signal is fragmented
    # but ignore contours that are too small or too vertical.
    picked_contours = []
    for c in sorted_contours[:5]:
        x, y, w, h = cv2.boundingRect(c)
        if w > width * 0.1: # Must span at least 10% of image
            picked_contours.append(c)
    
    if not picked_contours:
        # Fallback to the single widest if nothing passed the 10% filter
        picked_contours = [sorted_contours[0]]

    cv2.drawContours(signal_mask, picked_contours, -1, 255, thickness=1)

    # --- Step 4: Trace the Waveform ---
    # Now we scan column-wise but ONLY on the isolated signal mask.
    signal = []
    for col in range(width):
        white_pixels = np.where(signal_mask[:, col] > 0)[0]
        if len(white_pixels) > 0:
            # Use the mean of current column's signal pixels
            y_pos = np.mean(white_pixels)
            signal.append(height - y_pos)
        else:
            # Interpolate or hold last value
            if signal:
                signal.append(signal[-1])
            else:
                signal.append(height / 2)

    # --- Step 5: Final Cleanup ---
    signal = np.array(signal, dtype=float)
    signal = signal - np.mean(signal) # Zero-center

    # Downsample for visualization/performance
    if len(signal) > 2000:
        indices = np.linspace(0, len(signal)-1, 2000).astype(int)
        signal = signal[indices]

    return signal.tolist()
