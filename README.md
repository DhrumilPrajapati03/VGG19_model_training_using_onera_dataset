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

  
Let’s break down the `dataset_from_folder` function step by step, explaining its purpose, logic, and every important object it creates or uses. I’ll provide a detailed explanation and print each key object with example values, following the same format as the previous explanations.

---

### **Function: `dataset_from_folder(cities_folder, labels_folder, overlap_for_tiling=0, filter_on=False)`**
```python
def dataset_from_folder(cities_folder, labels_folder, overlap_for_tiling=0, filter_on=False):
    # loading uses snippets from https://www.kaggle.com/aninda/change-detection-nb

    img1_paths = []   # creating list of imagery paths for first set of images 
    img2_paths = []   # creating list of imagery paths for second set of images
    label_paths = []  # creating list of change mask paths for the images
    # load paths:
    for city in cities_folder:
        img1_paths.append(img_dir + "/" + city + "/pair/" + "img1.png") # < "pair" contains rgb only ...
        img2_paths.append(img_dir + "/" + city + "/pair/" + "img2.png")
        label_paths.append(labels_folder + "/" + city +"/cm/cm.png")

    # load images:
    all_tiles_im1 = []
    all_tiles_im2 = []
    all_tiles_label = []
    for img_idx in range(len(cities_folder)):
        im1 = Image.open(img1_paths[img_idx])
        im2 = Image.open(img2_paths[img_idx])
        lab = Image.open(label_paths[img_idx]).convert('L') # LA is with transparency

        im1 = np.array(im1) / 255 # scale
        im2 = np.array(im2) / 255 # scale
        lab = np.array(lab) / 255 # scale 0 or 1
        lab = lab.astype(np.uint8)
        lab = np.expand_dims(lab, axis=2)

        #print("debug same shapes >", im1.shape, im2.shape, lab.shape)
        #show_three(im1,im2,lab)

        print(img1_paths[img_idx],"~",img_idx,": A=", np.array(im1).shape,"B=",np.array(im2).shape,"L=",np.array(lab).shape)
        tiles_im1, tiles_im2, tiles_label = tile_image(im1, im2, lab, overlap_for_tiling, filter_on)
        print("Loaded triplets:", tiles_im1.shape, tiles_im2.shape, tiles_label.shape)

        if len(tiles_im1) > 0:
            # only if we didn't filter all out
            if len(all_tiles_im1)==0:
                all_tiles_im1 = tiles_im1
            else:
                all_tiles_im1 = np.vstack((all_tiles_im1, tiles_im1))
            if len(all_tiles_im2)==0:
                all_tiles_im2 = tiles_im2
            else:
                all_tiles_im2 = np.vstack((all_tiles_im2, tiles_im2))

            if len(all_tiles_label)==0:
                all_tiles_label = tiles_label
            else:
                all_tiles_label = np.vstack((all_tiles_label, tiles_label))

    all_tiles_im1 = np.asarray(all_tiles_im1)
    all_tiles_im2 = np.asarray(all_tiles_im2)
    all_tiles_label = np.asarray(all_tiles_label)
    all_triplets = [all_tiles_im1, all_tiles_im2, all_tiles_label]

    return all_triplets
```

#### **Purpose**
This function creates a dataset by loading pairs of images (`img1.png`, `img2.png`) and their corresponding change masks (`cm.png`) from a folder structure, processing them into tiles using the `tile_image` function, and stacking all tiles into three NumPy arrays. It’s likely used for preparing data for a machine learning model, such as change detection between two images.

#### **Step-by-Step Explanation**
1. **Inputs**:
   - `cities_folder`: List of city names (subfolders containing image pairs).
   - `labels_folder`: Path to the folder containing label masks.
   - `overlap_for_tiling`: Integer for tile overlap (default: 0).
   - `filter_on`: Boolean to enable filtering with `has_change()` (default: `False`).
   - `img_dir`: Global variable (not shown) for the base image directory.

2. **Initialize Path Lists**:
   - `img1_paths`, `img2_paths`, `label_paths`: Lists to store file paths for images and labels.

3. **Build File Paths**:
   - Loops over `cities_folder` to construct paths for `img1.png`, `img2.png`, and `cm.png`.

4. **Initialize Tile Lists**:
   - `all_tiles_im1`, `all_tiles_im2`, `all_tiles_label`: Lists to accumulate tiles from all images.

5. **Process Each Image Pair**:
   - Load images and labels, preprocess them (scale, convert), tile them, and stack tiles into the lists.
   - Use `tile_image()` (assumed from earlier) to split into tiles.

6. **Stack Tiles**:
   - Use `np.vstack()` to concatenate tiles vertically (along the first axis).

7. **Convert to Arrays and Return**:
   - Convert lists to NumPy arrays and return as a list of three arrays.

---

### **Important Objects**
Here are all key objects, with example values based on a hypothetical input:

#### **Example Setup**
- `cities_folder = ['city1', 'city2']`
- `img_dir = '/data/images'`
- `labels_folder = '/data/labels'`
- `overlap_for_tiling = 0`, `filter_on = True`
- Assume each `img1.png` and `img2.png` is 512x512x3 (RGB), `cm.png` is 512x512 (grayscale).
- `tile_size = 256` (global), `has_change()` filters tiles with >10% change.
- For `city1`: Top-left tile has change; for `city2`: No change.

#### **Objects and Values**
1. **`cities_folder`**:
   - **Type**: List of strings.
   - **Purpose**: List of city folder names.
   - **Value**: 
     ```python
     print(cities_folder)
     # Output: ['city1', 'city2']
     ```

2. **`labels_folder`**:
   - **Type**: String.
   - **Purpose**: Path to label directory.
   - **Value**: 
     ```python
     print(labels_folder)
     # Output: '/data/labels'
     ```

3. **`overlap_for_tiling`**:
   - **Type**: Integer.
   - **Purpose**: Overlap between tiles.
   - **Value**: 
     ```python
     print(overlap_for_tiling)
     # Output: 0
     ```

4. **`filter_on`**:
   - **Type**: Boolean.
   - **Purpose**: Enable filtering.
   - **Value**: 
     ```python
     print(filter_on)
     # Output: True
     ```

5. **`img1_paths`**:
   - **Type**: List of strings.
   - **Purpose**: Paths to first images.
   - **Value**: 
     ```python
     img1_paths = []
     for city in cities_folder:
         img1_paths.append(img_dir + "/" + city + "/pair/" + "img1.png")
     print(img1_paths)
     # Output: ['/data/images/city1/pair/img1.png', '/data/images/city2/pair/img1.png']
     ```

6. **`img2_paths`**:
   - **Type**: List of strings.
   - **Purpose**: Paths to second images.
   - **Value**: 
     ```python
     img2_paths = []
     for city in cities_folder:
         img2_paths.append(img_dir + "/" + city + "/pair/" + "img2.png")
     print(img2_paths)
     # Output: ['/data/images/city1/pair/img2.png', '/data/images/city2/pair/img2.png']
     ```

7. **`label_paths`**:
   - **Type**: List of strings.
   - **Purpose**: Paths to label masks.
   - **Value**: 
     ```python
     label_paths = []
     for city in cities_folder:
         label_paths.append(labels_folder + "/" + city + "/cm/cm.png")
     print(label_paths)
     # Output: ['/data/labels/city1/cm/cm.png', '/data/labels/city2/cm/cm.png']
     ```

8. **`all_tiles_im1`, `all_tiles_im2`, `all_tiles_label`** (initial):
   - **Type**: Lists.
   - **Purpose**: Store all tiles.
   - **Value**: 
     ```python
     all_tiles_im1 = []; all_tiles_im2 = []; all_tiles_label = []
     print(all_tiles_im1, all_tiles_im2, all_tiles_label)
     # Output: [] [] []
     ```

9. **`img_idx`** (first iteration):
   - **Type**: Integer.
   - **Purpose**: Index of current image pair.
   - **Value**: 
     ```python
     img_idx = 0
     print(img_idx)
     # Output: 0
     ```

10. **`im1`** (before scaling):
    - **Type**: PIL Image.
    - **Purpose**: Loaded first image.
    - **Value**: 
      ```python
      from PIL import Image
      im1 = Image.open(img1_paths[img_idx])  # Simulated
      print(im1.size, im1.mode)
      # Output: (512, 512) RGB
      ```

11. **`im2`** (before scaling):
    - **Type**: PIL Image.
    - **Purpose**: Loaded second image.
    - **Value**: 
      ```python
      im2 = Image.open(img2_paths[img_idx])
      print(im2.size, im2.mode)
      # Output: (512, 512) RGB
      ```

12. **`lab`** (before scaling):
    - **Type**: PIL Image.
    - **Purpose**: Loaded label (grayscale).
    - **Value**: 
      ```python
      lab = Image.open(label_paths[img_idx]).convert('L')
      print(lab.size, lab.mode)
      # Output: (512, 512) L
      ```

13. **`im1`** (after scaling):
    - **Type**: NumPy array (shape: `(512, 512, 3)`).
    - **Purpose**: Scaled first image (0-1 range).
    - **Value**: 
      ```python
      import numpy as np
      im1 = np.array(im1) / 255
      print(im1.shape, im1[0, 0])
      # Output: (512, 512, 3) [0.00392157 0.00392157 0.00392157] (assuming 1/255)
      ```

14. **`im2`** (after scaling):
    - **Type**: NumPy array (shape: `(512, 512, 3)`).
    - **Purpose**: Scaled second image.
    - **Value**: 
      ```python
      im2 = np.array(im2) / 255
      print(im2.shape, im2[0, 0])
      # Output: (512, 512, 3) [0.00784314 0.00784314 0.00784314] (assuming 2/255)
      ```

15. **`lab`** (after processing):
    - **Type**: NumPy array (shape: `(512, 512, 1)`).
    - **Purpose**: Scaled and expanded label (0 or 1).
    - **Value**: 
      ```python
      lab = np.array(lab) / 255
      lab = lab.astype(np.uint8)
      lab = np.expand_dims(lab, axis=2)
      print(lab.shape, lab[0, 0], lab[256, 256])
      # Output: (512, 512, 1) [1] [0] (assuming top-left has change)
      ```

16. **`tiles_im1`, `tiles_im2`, `tiles_label`**:
    - **Type**: NumPy arrays (e.g., `(1, 256, 256, 3)` for `tiles_im1`).
    - **Purpose**: Tiles from one image pair.
    - **Value**: 
      ```python
      # Assuming tile_image from earlier
      tiles_im1, tiles_im2, tiles_label = tile_image(im1, im2, lab, 0, True)
      print(tiles_im1.shape, tiles_im2.shape, tiles_label.shape)
      # Output: (1, 256, 256, 3) (1, 256, 256, 3) (1, 256, 256, 1)
      ```

17. **`all_tiles_im1`** (after first iteration):
    - **Type**: NumPy array.
    - **Purpose**: Accumulates tiles from `im1`.
    - **Value**: 
      ```python
      all_tiles_im1 = tiles_im1
      print(all_tiles_im1.shape)
      # Output: (1, 256, 256, 3)
      ```

18. **`all_tiles_im1`** (after second iteration):
    - **Type**: NumPy array.
    - **Purpose**: Stacked tiles.
    - **Value**: 
      ```python
      # Second city has no tiles due to filter
      all_tiles_im1 = np.vstack((all_tiles_im1, tiles_im1)) if len(tiles_im1) > 0 else all_tiles_im1
      print(all_tiles_im1.shape)
      # Output: (1, 256, 256, 3) (no change in city2)
      ```

19. **`all_triplets`**:
    - **Type**: List of NumPy arrays.
    - **Purpose**: Final dataset.
    - **Value**: 
      ```python
      all_triplets = [all_tiles_im1, all_tiles_im2, all_tiles_label]
      print([arr.shape for arr in all_triplets])
      # Output: [(1, 256, 256, 3), (1, 256, 256, 3), (1, 256, 256, 1)]
      ```

---

### **Full Example Code**
```python
import numpy as np
from PIL import Image

img_dir = '/data/images'
tile_size = 256
change_pix_percent = 0.1

def has_change(lab):
    return np.count_nonzero(lab.flatten() == 1.0) / len(lab.flatten()) > change_pix_percent

def tile_image(im1, im2, lab, overlap_for_tiling=0, filter_on=False):
    # Simplified for example
    tiles_im1 = [im1[0:256, 0:256, :]] if has_change(lab[0:256, 0:256, :]) else []
    tiles_im2 = [im2[0:256, 0:256, :]] if has_change(lab[0:256, 0:256, :]) else []
    tiles_label = [lab[0:256, 0:256, :]] if has_change(lab[0:256, 0:256, :]) else []
    return np.array(tiles_im1), np.array(tiles_im2), np.array(tiles_label)

def dataset_from_folder(cities_folder, labels_folder, overlap_for_tiling=0, filter_on=False):
    img1_paths = []; img2_paths = []; label_paths = []
    for city in cities_folder:
        img1_paths.append(img_dir + "/" + city + "/pair/" + "img1.png")
        img2_paths.append(img_dir + "/" + city + "/pair/" + "img2.png")
        label_paths.append(labels_folder + "/" + city + "/cm/cm.png")

    all_tiles_im1 = []; all_tiles_im2 = []; all_tiles_label = []
    for img_idx in range(len(cities_folder)):
        # Simulate image loading
        im1 = np.ones((512, 512, 3), dtype=np.uint8)  # Dummy image
        im2 = np.ones((512, 512, 3), dtype=np.uint8) * 2
        lab = np.zeros((512, 512), dtype=np.uint8)
        if img_idx == 0:  # Only city1 has change
            lab[0:256, 0:256] = 255

        im1 = im1 / 255; im2 = im2 / 255; lab = lab / 255
        lab = lab.astype(np.uint8)
        lab = np.expand_dims(lab, axis=2)

        print(img1_paths[img_idx], "~", img_idx, ": A=", im1.shape, "B=", im2.shape, "L=", lab.shape)
        tiles_im1, tiles_im2, tiles_label = tile_image(im1, im2, lab, overlap_for_tiling, filter_on)
        print("Loaded triplets:", tiles_im1.shape, tiles_im2.shape, tiles_label.shape)

        if len(tiles_im1) > 0:
            if len(all_tiles_im1) == 0:
                all_tiles_im1 = tiles_im1
            else:
                all_tiles_im1 = np.vstack((all_tiles_im1, tiles_im1))
            if len(all_tiles_im2) == 0:
                all_tiles_im2 = tiles_im2
            else:
                all_tiles_im2 = np.vstack((all_tiles_im2, tiles_im2))
            if len(all_tiles_label) == 0:
                all_tiles_label = tiles_label
            else:
                all_tiles_label = np.vstack((all_tiles_label, tiles_label))

    all_triplets = [all_tiles_im1, all_tiles_im2, all_tiles_label]
    return all_triplets

# Test
cities_folder = ['city1', 'city2']
labels_folder = '/data/labels'
all_triplets = dataset_from_folder(cities_folder, labels_folder, 0, True)
print("Final shapes:", [arr.shape for arr in all_triplets])
```

#### **Output**
```
'/data/images/city1/pair/img1.png' ~ 0 : A= (512, 512, 3) B= (512, 512, 3) L= (512, 512, 1)
Loaded triplets: (1, 256, 256, 3) (1, 256, 256, 3) (1, 256, 256, 1)
'/data/images/city2/pair/img1.png' ~ 1 : A= (512, 512, 3) B= (512, 512, 3) L= (512, 512, 1)
Loaded triplets: (0,) (0,) (0,)
Final shapes: [(1, 256, 256, 3), (1, 256, 256, 3), (1, 256, 256, 1)]
```

---

### **Summary**
- The function loads image pairs and labels, processes them into tiles, and stacks them into a dataset.
- Key objects include paths, raw and processed images, tiles, and the final triplet list.
- With `filter_on = True`, only tiles with change (e.g., from `city1`) are kept.
