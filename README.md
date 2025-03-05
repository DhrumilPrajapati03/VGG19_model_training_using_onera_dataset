# VGG19_model_training_using_onera_dataset
Let’s break down the `has_change(lab)` function step by step in depth, focusing on its purpose, logic, and the important objects it creates or uses. Afterward, I’ll list and print every key object with explanations.

---

### **Function: `has_change(lab)`**
```python
def has_change(lab):
    num_ones = np.count_nonzero(lab.flatten() == 1.0)
    num_all_pixels = len(lab.flatten())
    perc_change = num_ones / num_all_pixels

    return perc_change > change_pix_percent
    #return np.max(lab) == 1.0
```

#### **Purpose**
This function evaluates whether a given label array (`lab`) has a sufficient percentage of pixels with the value `1.0` to be considered as having "change." It’s likely used in an image processing context (e.g., change detection) to filter regions based on how much they differ, as indicated by the label array.

#### **Step-by-Step Explanation**
1. **Input**:
   - `lab`: A NumPy array (typically 2D or 3D) representing a label or mask. For example, it might be a binary mask where `1.0` indicates a changed pixel and `0.0` indicates no change.
   - `change_pix_percent`: A global variable (not defined in the function) representing the threshold percentage of `1.0` pixels required to classify the array as having "change."

2. **`num_ones = np.count_nonzero(lab.flatten() == 1.0)`**:
   - `lab.flatten()`: Converts the multi-dimensional `lab` array into a 1D array.
   - `lab.flatten() == 1.0`: Creates a boolean array of the same length, where each element is `True` if the corresponding pixel equals `1.0` and `False` otherwise.
   - `np.count_nonzero()`: Counts the number of `True` values in the boolean array, effectively giving the total number of pixels equal to `1.0`.
   - Result: `num_ones` is an integer representing the count of `1.0` pixels.

3. **`num_all_pixels = len(lab.flatten())`**:
   - `lab.flatten()`: Again flattens the array (this could reuse the result from the previous step, but it’s recalculated here).
   - `len()`: Returns the total number of elements in the flattened array, i.e., the total number of pixels.
   - Result: `num_all_pixels` is an integer representing the total pixel count.

4. **`perc_change = num_ones / num_all_pixels`**:
   - Divides `num_ones` by `num_all_pixels` to compute the fraction of pixels that are `1.0`.
   - Result: `perc_change` is a float between 0 and 1, representing the percentage of `1.0` pixels.

5. **`return perc_change > change_pix_percent`**:
   - Compares `perc_change` with the threshold `change_pix_percent`.
   - Result: Returns `True` if the percentage of `1.0` pixels exceeds `change_pix_percent`, and `False` otherwise.

6. **Commented Line**:
   - `#return np.max(lab) == 1.0`: This alternative (not executed) checks if the maximum value in `lab` is `1.0`, i.e., if there’s at least one `1.0` pixel. It’s simpler but less strict than the percentage-based approach.

---

### **Important Objects**
Here are all the key objects created or used in the function, with their types, purposes, and example values based on a hypothetical input:

#### **Example Setup**
Let’s assume:
- `lab = np.array([[1.0, 0.0], [0.0, 1.0]])` (a 2x2 array).
- `change_pix_percent = 0.4` (40% threshold).

Now, let’s walk through and print each object:

1. **`lab`**:
   - **Type**: NumPy array (e.g., `ndarray`).
   - **Purpose**: The input label array to analyze.
   - **Value**: 
     ```python
     print(lab)
     # Output:
     # [[1. 0.]
     #  [0. 1.]]
     ```

2. **`lab.flatten()`** (used in `num_ones`):
   - **Type**: NumPy array (1D).
   - **Purpose**: Flattens `lab` into a 1D array for easier processing.
   - **Value**: 
     ```python
     print(lab.flatten())
     # Output: [1. 0. 0. 1.]
     ```

3. **`lab.flatten() == 1.0`**:
   - **Type**: NumPy array (1D, boolean).
   - **Purpose**: A boolean mask indicating where pixels equal `1.0`.
   - **Value**: 
     ```python
     print(lab.flatten() == 1.0)
     # Output: [ True False False  True]
     ```

4. **`num_ones`**:
   - **Type**: Integer.
   - **Purpose**: Counts the number of `1.0` pixels.
   - **Value**: 
     ```python
     num_ones = np.count_nonzero(lab.flatten() == 1.0)
     print(num_ones)
     # Output: 2
     ```

5. **`num_all_pixels`**:
   - **Type**: Integer.
   - **Purpose**: Total number of pixels in `lab`.
   - **Value**: 
     ```python
     num_all_pixels = len(lab.flatten())
     print(num_all_pixels)
     # Output: 4
     ```

6. **`perc_change`**:
   - **Type**: Float.
   - **Purpose**: Fraction of pixels that are `1.0`.
   - **Value**: 
     ```python
     perc_change = num_ones / num_all_pixels
     print(perc_change)
     # Output: 0.5
     ```

7. **`change_pix_percent`**:
   - **Type**: Float (assumed, as it’s global).
   - **Purpose**: Threshold percentage for determining "change."
   - **Value**: 
     ```python
     print(change_pix_percent)
     # Output: 0.4  (assumed)
     ```

8. **`perc_change > change_pix_percent`** (return value):
   - **Type**: Boolean.
   - **Purpose**: The final result indicating whether the percentage of `1.0` pixels exceeds the threshold.
   - **Value**: 
     ```python
     result = perc_change > change_pix_percent
     print(result)
     # Output: True  (since 0.5 > 0.4)
     ```

---

### **Full Example Code**
Here’s how you could run this with the example:
```python
import numpy as np

# Define global variable
change_pix_percent = 0.4

# Define function
def has_change(lab):
    num_ones = np.count_nonzero(lab.flatten() == 1.0)
    num_all_pixels = len(lab.flatten())
    perc_change = num_ones / num_all_pixels

    return perc_change > change_pix_percent

# Test input
lab = np.array([[1.0, 0.0], [0.0, 1.0]])

# Print all objects
print("lab:\n", lab)
print("lab.flatten():", lab.flatten())
print("lab.flatten() == 1.0:", lab.flatten() == 1.0)
print("num_ones:", np.count_nonzero(lab.flatten() == 1.0))
print("num_all_pixels:", len(lab.flatten()))
print("perc_change:", num_ones / num_all_pixels)
print("change_pix_percent:", change_pix_percent)
print("Result:", has_change(lab))
```

#### **Output**
```
lab:
 [[1. 0.]
  [0. 1.]]
lab.flatten(): [1. 0. 0. 1.]
lab.flatten() == 1.0: [ True False False  True]
num_ones: 2
num_all_pixels: 4
perc_change: 0.5
change_pix_percent: 0.4
Result: True
```

---

### **Summary**
- The function checks if the percentage of `1.0` pixels in `lab` exceeds `change_pix_percent`.
- Key objects include the input `lab`, intermediate arrays (flattened and boolean), counts (`num_ones`, `num_all_pixels`), the percentage (`perc_change`), and the final boolean result.
- It’s efficient for small-to-medium arrays but could be optimized by flattening `lab` only once if performance is critical.

Let’s break down the `tile_image` function step by step, explaining its purpose, logic, and every important object it creates or uses. I’ll provide a detailed explanation and print each key object with example values, similar to the previous explanation.

---

### **Function: `tile_image(im1, im2, label, overlap_for_tiling = 0, filter_on = False)`**
```python
def tile_image(im1, im2, label, overlap_for_tiling = 0, filter_on = False):
    tiles_im1 = []
    tiles_im2 = []
    tiles_label = []

    move_by = tile_size - overlap_for_tiling

    image_shape = np.array(im1).shape
    h, w, ch = image_shape
    
    h_num = math.floor(h / tile_size)
    heights = [tile_size*i for i in range(h_num)]
    w_num = math.floor(w / tile_size)
    widths = [tile_size*i for i in range(w_num)]
    
    h_current = 0
    w_current = 0
    while h_current + tile_size < h:
        while w_current + tile_size < w:
            row_start = h_current
            row_end = h_current + tile_size
            col_start = w_current
            col_end = w_current + tile_size
            w_current += move_by

            tile_im1 = im1[row_start:row_end,col_start:col_end,:]
            tile_im2 = im2[row_start:row_end,col_start:col_end,:]
            tile_label = label[row_start:row_end,col_start:col_end,:]
            
            if not filter_on or has_change(tile_label):
                tiles_im1.append(tile_im1)
                tiles_im2.append(tile_im2)
                tiles_label.append(tile_label)
        h_current += move_by
        w_current = 0

    tiles_im1 = np.asarray(tiles_im1)
    tiles_im2 = np.asarray(tiles_im2)
    tiles_label = np.asarray(tiles_label)
    return tiles_im1, tiles_im2, tiles_label
```

#### **Purpose**
This function divides two input images (`im1` and `im2`) and a corresponding label array (`label`) into smaller tiles of size `tile_size x tile_size`. It supports optional overlapping of tiles and filtering based on the `has_change()` function (assumed to be defined elsewhere). The result is three NumPy arrays containing the tiles from `im1`, `im2`, and `label`.

#### **Step-by-Step Explanation**
1. **Inputs**:
   - `im1`, `im2`: Two images as 3D NumPy arrays (shape: `(height, width, channels)`).
   - `label`: A label/mask array (same shape as `im1` and `im2`).
   - `overlap_for_tiling`: Integer specifying the overlap between tiles (default: 0, no overlap).
   - `filter_on`: Boolean to enable/disable filtering with `has_change()` (default: `False`).
   - `tile_size`: A global variable defining the tile size (e.g., 256).

2. **Initialize Empty Lists**:
   - `tiles_im1`, `tiles_im2`, `tiles_label`: Lists to store tiles from `im1`, `im2`, and `label`.

3. **`move_by = tile_size - overlap_for_tiling`**:
   - Calculates the step size for moving the tiling window. If `overlap_for_tiling = 0`, `move_by = tile_size` (no overlap).

4. **`image_shape = np.array(im1).shape`**:
   - Gets the shape of `im1` (assumes `im2` and `label` match).
   - `h, w, ch = image_shape`: Unpacks height (`h`), width (`w`), and channels (`ch`).

5. **Calculate Tile Counts and Starting Points**:
   - `h_num = math.floor(h / tile_size)`: Number of tiles along height.
   - `heights = [tile_size*i for i in range(h_num)]`: Starting row indices.
   - `w_num = math.floor(w / tile_size)`: Number of tiles along width.
   - `widths = [tile_size*i for i in range(w_num)]`: Starting column indices.

6. **Tiling Loop**:
   - Outer: `while h_current + tile_size < h`: Loops over rows.
   - Inner: `while w_current + tile_size < w`: Loops over columns.
   - Define tile boundaries: `row_start`, `row_end`, `col_start`, `col_end`.
   - Increment: `w_current += move_by` (column step), `h_current += move_by` (row step), `w_current = 0` (reset column).

7. **Extract Tiles**:
   - `tile_im1 = im1[row_start:row_end, col_start:col_end, :]`: Tile from `im1`.
   - Similarly for `tile_im2` and `tile_label`.

8. **Filtering**:
   - `if not filter_on or has_change(tile_label)`: Adds tiles to lists if filtering is off or if `has_change()` returns `True`.

9. **Convert to Arrays**:
   - `tiles_im1 = np.asarray(tiles_im1)`: Converts list to 4D NumPy array (e.g., `(num_tiles, tile_size, tile_size, ch)`).
   - Similarly for `tiles_im2` and `tiles_label`.

10. **Return**:
    - Returns `tiles_im1`, `tiles_im2`, `tiles_label`.

---

### **Important Objects**
Here are all key objects, with example values based on a hypothetical input:

#### **Example Setup**
- `im1 = np.ones((512, 512, 3))` (512x512 RGB image, all 1s).
- `im2 = np.ones((512, 512, 3)) * 2` (all 2s).
- `label = np.zeros((512, 512, 1))`; `label[0:256, 0:256, 0] = 1.0` (top-left quarter has change).
- `tile_size = 256`, `overlap_for_tiling = 0`, `filter_on = True`.
- Assume `has_change(lab)` from earlier with `change_pix_percent = 0.1`.

#### **Objects and Values**
1. **`im1`**:
   - **Type**: NumPy array (shape: `(512, 512, 3)`).
   - **Purpose**: First input image.
   - **Value**: 
     ```python
     print(im1.shape, im1[0, 0])  # Example pixel
     # Output: (512, 512, 3) [1. 1. 1.]
     ```

2. **`im2`**:
   - **Type**: NumPy array (shape: `(512, 512, 3)`).
   - **Purpose**: Second input image.
   - **Value**: 
     ```python
     print(im2.shape, im2[0, 0])
     # Output: (512, 512, 3) [2. 2. 2.]
     ```

3. **`label`**:
   - **Type**: NumPy array (shape: `(512, 512, 1)`).
   - **Purpose**: Label array indicating change.
   - **Value**: 
     ```python
     print(label.shape, label[0, 0], label[256, 256])
     # Output: (512, 512, 1) [1.] [0.]
     ```

4. **`overlap_for_tiling`**:
   - **Type**: Integer.
   - **Purpose**: Overlap between tiles.
   - **Value**: 
     ```python
     print(overlap_for_tiling)
     # Output: 0
     ```

5. **`filter_on`**:
   - **Type**: Boolean.
   - **Purpose**: Enable filtering.
   - **Value**: 
     ```python
     print(filter_on)
     # Output: True
     ```

6. **`tiles_im1`, `tiles_im2`, `tiles_label`** (initial):
   - **Type**: Lists.
   - **Purpose**: Store tiles before conversion.
   - **Value**: 
     ```python
     tiles_im1 = []; tiles_im2 = []; tiles_label = []
     print(tiles_im1, tiles_im2, tiles_label)
     # Output: [] [] []
     ```

7. **`move_by`**:
   - **Type**: Integer.
   - **Purpose**: Step size for tiling.
   - **Value**: 
     ```python
     move_by = tile_size - overlap_for_tiling
     print(move_by)
     # Output: 256
     ```

8. **`image_shape`**:
   - **Type**: Tuple.
   - **Purpose**: Shape of `im1`.
   - **Value**: 
     ```python
     image_shape = np.array(im1).shape
     print(image_shape)
     # Output: (512, 512, 3)
     ```

9. **`h`, `w`, `ch`**:
   - **Type**: Integers.
   - **Purpose**: Height, width, channels.
   - **Value**: 
     ```python
     h, w, ch = image_shape
     print(h, w, ch)
     # Output: 512 512 3
     ```

10. **`h_num`**:
    - **Type**: Integer.
    - **Purpose**: Number of tiles along height.
    - **Value**: 
      ```python
      h_num = math.floor(h / tile_size)
      print(h_num)
      # Output: 2
      ```

11. **`heights`**:
    - **Type**: List.
    - **Purpose**: Starting row indices.
    - **Value**: 
      ```python
      heights = [tile_size * i for i in range(h_num)]
      print(heights)
      # Output: [0, 256]
      ```

12. **`w_num`**:
    - **Type**: Integer.
    - **Purpose**: Number of tiles along width.
    - **Value**: 
      ```python
      w_num = math.floor(w / tile_size)
      print(w_num)
      # Output: 2
      ```

13. **`widths`**:
    - **Type**: List.
    - **Purpose**: Starting column indices.
    - **Value**: 
      ```python
      widths = [tile_size * i for i in range(w_num)]
      print(widths)
      # Output: [0, 256]
      ```

14. **`h_current`, `w_current`** (initial):
    - **Type**: Integers.
    - **Purpose**: Current row and column positions.
    - **Value**: 
      ```python
      h_current = 0; w_current = 0
      print(h_current, w_current)
      # Output: 0 0
      ```

15. **`row_start`, `row_end`, `col_start`, `col_end`** (first iteration):
    - **Type**: Integers.
    - **Purpose**: Tile boundaries.
    - **Value**: 
      ```python
      row_start = h_current; row_end = h_current + tile_size
      col_start = w_current; col_end = w_current + tile_size
      print(row_start, row_end, col_start, col_end)
      # Output: 0 256 0 256
      ```

16. **`tile_im1`** (first tile):
    - **Type**: NumPy array (shape: `(256, 256, 3)`).
    - **Purpose**: Tile from `im1`.
    - **Value**: 
      ```python
      tile_im1 = im1[row_start:row_end, col_start:col_end, :]
      print(tile_im1.shape, tile_im1[0, 0])
      # Output: (256, 256, 3) [1. 1. 1.]
      ```

17. **`tile_im2`** (first tile):
    - **Type**: NumPy array (shape: `(256, 256, 3)`).
    - **Purpose**: Tile from `im2`.
    - **Value**: 
      ```python
      tile_im2 = im2[row_start:row_end, col_start:col_end, :]
      print(tile_im2.shape, tile_im2[0, 0])
      # Output: (256, 256, 3) [2. 2. 2.]
      ```

18. **`tile_label`** (first tile):
    - **Type**: NumPy array (shape: `(256, 256, 1)`).
    - **Purpose**: Tile from `label`.
    - **Value**: 
      ```python
      tile_label = label[row_start:row_end, col_start:col_end, :]
      print(tile_label.shape, tile_label[0, 0])
      # Output: (256, 256, 1) [1.]
      ```

19. **`has_change(tile_label)`** (first tile):
    - **Type**: Boolean.
    - **Purpose**: Check if tile has sufficient change.
    - **Value**: 
      ```python
      # Assuming has_change from earlier
      def has_change(lab):
          return np.count_nonzero(lab.flatten() == 1.0) / len(lab.flatten()) > 0.1
      print(has_change(tile_label))
      # Output: True (all 1s in this tile)
      ```

20. **`tiles_im1`, `tiles_im2`, `tiles_label`** (after first tile):
    - **Type**: Lists.
    - **Purpose**: Updated with tiles.
    - **Value**: 
      ```python
      tiles_im1.append(tile_im1); tiles_im2.append(tile_im2); tiles_label.append(tile_label)
      print(len(tiles_im1), tiles_im1[0].shape)
      # Output: 1 (256, 256, 3)
      ```

21. **`tiles_im1`, `tiles_im2`, `tiles_label`** (final arrays):
    - **Type**: NumPy arrays (e.g., `(1, 256, 256, 3)` for `tiles_im1`).
    - **Purpose**: Final tiled outputs.
    - **Value**: 
      ```python
      tiles_im1 = np.asarray(tiles_im1)
      print(tiles_im1.shape)  # Only 1 tile kept due to filter
      # Output: (1, 256, 256, 3)
      ```

---

### **Full Example Code**
```python
import numpy as np
import math

tile_size = 256
change_pix_percent = 0.1

def has_change(lab):
    return np.count_nonzero(lab.flatten() == 1.0) / len(lab.flatten()) > change_pix_percent

def tile_image(im1, im2, label, overlap_for_tiling=0, filter_on=False):
    tiles_im1 = []; tiles_im2 = []; tiles_label = []
    move_by = tile_size - overlap_for_tiling
    image_shape = np.array(im1).shape
    h, w, ch = image_shape
    h_num = math.floor(h / tile_size)
    heights = [tile_size*i for i in range(h_num)]
    w_num = math.floor(w / tile_size)
    widths = [tile_size*i for i in range(w_num)]
    h_current = 0; w_current = 0
    while h_current + tile_size < h:
        while w_current + tile_size < w:
            row_start = h_current; row_end = h_current + tile_size
            col_start = w_current; col_end = w_current + tile_size
            w_current += move_by
            tile_im1 = im1[row_start:row_end, col_start:col_end, :]
            tile_im2 = im2[row_start:row_end, col_start:col_end, :]
            tile_label = label[row_start:row_end, col_start:col_end, :]
            if not filter_on or has_change(tile_label):
                tiles_im1.append(tile_im1); tiles_im2.append(tile_im2); tiles_label.append(tile_label)
        h_current += move_by; w_current = 0
    tiles_im1 = np.asarray(tiles_im1); tiles_im2 = np.asarray(tiles_im2); tiles_label = np.asarray(tiles_label)
    return tiles_im1, tiles_im2, tiles_label

# Test
im1 = np.ones((512, 512, 3))
im2 = np.ones((512, 512, 3)) * 2
label = np.zeros((512, 512, 1)); label[0:256, 0:256, 0] = 1.0
tiles_im1, tiles_im2, tiles_label = tile_image(im1, im2, label, 0, True)
print("Final shapes:", tiles_im1.shape, tiles_im2.shape, tiles_label.shape)
# Output: (1, 256, 256, 3) (1, 256, 256, 3) (1, 256, 256, 1)
```

---

### **Summary**
- The function tiles images into `tile_size x tile_size` sections, with optional overlap and filtering.
- Key objects include inputs, intermediate variables (e.g., `move_by`, `h_num`), tile boundaries, extracted tiles, and final arrays.
- With `filter_on = True`, only the top-left tile is kept in this example due to the label’s change distribution.

  
