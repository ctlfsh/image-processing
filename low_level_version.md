# Image Segmentation Approaches: OpenCV/NumPy vs. scikit-image

This document outlines two different approaches for implementing the signal detection script, highlighting the trade-offs between using a pure `OpenCV/NumPy` stack versus incorporating the `scikit-image` library.

## Approach 1: Using `opencv-python` and `numpy`

This approach relies solely on OpenCV and NumPy for all image processing tasks.

### Implementation Details

1.  **Image Loading and Preprocessing:** Use `cv2.imread` to load the image and `cv2.cvtColor` to convert it to the HSV color space.
2.  **Segmentation:** Use `cv2.inRange` to perform color-based thresholding, creating a binary mask of the signal.
3.  **Object Detection (Connected Components):** Use `cv2.connectedComponentsWithStats` to find distinct, connected regions (the signals) in the binary mask. This function returns:
    *   The number of labels.
    *   A `labels` matrix where each region is marked with a different integer.
    *   A `stats` matrix containing bounding box information (x, y, width, height) and the area for each labeled region.
    *   A `centroids` matrix.
4.  **Symmetry Calculation:** For each region identified by `cv2.connectedComponentsWithStats`, extract the bounding box, split the corresponding region of the mask in half, flip one half, and compare it to the other half using NumPy for pixel-wise comparison.

### Pros & Cons

*   **Pros:**
    *   Fewer dependencies (only `opencv-python` and `numpy`).
    *   OpenCV is highly optimized for performance.
*   **Cons:**
    *   The code can be slightly more verbose, as you need to manually iterate through the `stats` array returned by `cv2.connectedComponentsWithStats`.

## Approach 2: Using `scikit-image`

This approach leverages `scikit-image` for a more high-level, descriptive approach to object analysis.

### Implementation Details

1.  **Image Loading and Preprocessing:** Same as the OpenCV approach (`cv2.imread`, `cv2.cvtColor`, `cv2.inRange`). The binary mask is then passed to `scikit-image`.
2.  **Object Detection (Connected Components):**
    *   Use `skimage.measure.label` to label connected regions in the binary mask. This is analogous to the first part of `cv2.connectedComponentsWithStats`.
    *   Use `skimage.measure.regionprops` on the labeled image. This function returns a list of `RegionProperties` objects, where each object corresponds to one labeled region.
3.  **Symmetry Calculation:** Iterate through the `RegionProperties` objects. Each object provides convenient access to properties like `bbox`, which can be used to extract the region from the mask and perform the symmetry calculation, similar to the OpenCV method.

### Pros & Cons

*   **Pros:**
    *   **Readability:** The code is often more readable and "Pythonic". The `regionprops` function provides a clean, object-oriented way to access properties of each detected region (e.g., `region.area`, `region.bbox`, `region.centroid`).
    *   **Rich Features:** `regionprops` can calculate a large number of properties out-of-the-box (e.g., eccentricity, orientation, convex area), which can be useful for more advanced filtering or analysis in the future.
    *   **Maintainability:** For developers familiar with the SciPy ecosystem, this approach is very standard and easy to extend.
*   **Cons:**
    *   **Additional Dependency:** Adds `scikit-image` to the project's dependencies, which, as we saw, can sometimes lead to installation challenges on newer Python versions.

## Summary of Differences

| Feature                  | `opencv-python` + `numpy`                               | `scikit-image`                                                              |
| ------------------------ | ------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Dependencies**         | Minimal (just the two libraries)                        | Adds `scikit-image`                                                         |
| **API Style**            | More procedural; functions return matrices of stats.    | More object-oriented; `regionprops` returns a list of objects.              |
| **Code Readability**     | Can be less intuitive for complex property extraction.  | Generally higher, especially when accessing multiple region properties.      |
| **Feature Set**          | Provides core stats (bounding box, area, centroid).     | Provides a very extensive set of pre-computed region properties.            |
| **Ease of Maintenance**  | Straightforward if you're comfortable with NumPy slicing. | Very high, especially if you need to add more complex analysis later.       |

For the current task, both approaches are perfectly viable. The `scikit-image` approach is arguably cleaner and more extensible, making it a better choice if the dependency can be resolved. The pure OpenCV/NumPy approach is a robust fallback that avoids the dependency issue entirely.
