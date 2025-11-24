
import cv2
import numpy as np
from skimage.measure import label, regionprops
import os

def calculate_vertical_symmetry(mask):
    """
    Calculates the vertical symmetry of a binary mask.

    The score is the intersection over union (IoU) of the left half
    and the flipped right half of the mask. A score of 1.0 means
    perfect symmetry.
    """
    height, width = mask.shape
    if width < 2:
        return 0.0

    mid_point = width // 2
    left_half = mask[:, :mid_point]
    right_half = mask[:, mid_point:]

    # Make the halves equal width
    if left_half.shape[1] > right_half.shape[1]:
        left_half = left_half[:, :right_half.shape[1]]
    elif right_half.shape[1] > left_half.shape[1]:
        right_half = right_half[:, :left_half.shape[1]]

    flipped_right_half = np.fliplr(right_half)

    intersection = np.sum(np.logical_and(left_half, flipped_right_half))
    union = np.sum(np.logical_or(left_half, flipped_right_half))

    if union == 0:
        return 1.0  # Empty mask is symmetric

    return intersection / union

def find_signals(image_path, min_area=100):
    """
    Finds signals in an image, calculates their vertical symmetry,
    and returns a list of detections.
    """
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"

    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Could not read image from {image_path}"

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # We assume the signal is brighter and more saturated than the background.
    # The hue can vary, so we focus on saturation and value.
    # This range can be tweaked for better performance on different datasets.
    lower_bound = np.array([0, 50, 150])
    upper_bound = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Use morphology to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    detections = []
    for region in regions:
        if region.area >= min_area:
            minr, minc, maxr, maxc = region.bbox
            
            # Create a sub-mask for the specific region
            region_mask = (labeled_mask[minr:maxr, minc:maxc] == region.label).astype(np.uint8) * 255

            symmetry_score = calculate_vertical_symmetry(region_mask)

            detections.append({
                "location": (int(region.centroid[1]), int(region.centroid[0])),
                "bounding_box": (minc, minr, maxc - minc, maxr - minr),
                "symmetry_score": symmetry_score,
                "area": region.area
            })

    return detections

if __name__ == "__main__":
    image_files = ["s1.png", "s2.png", "s3.png", "s4.png"]

    for image_file in image_files:
        print(f"--- Processing {image_file} ---")
        results = find_signals(image_file)
        if isinstance(results, str):
            print(results)
            continue
        
        if not results:
            print("No signals detected.")
        else:
            # Sort results by symmetry score in descending order
            sorted_results = sorted(results, key=lambda x: x["symmetry_score"], reverse=True)
            for i, res in enumerate(sorted_results):
                print(
                    f"  Signal {i+1}: "
                    f"Location=(x:{res['location'][0]}, y:{res['location'][1]}), "
                    f"Symmetry Score={res['symmetry_score']:.3f}, "
                    f"Area={res['area']}"
                )
        print("\n")
