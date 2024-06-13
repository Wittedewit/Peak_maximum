# README

## Overview

This project involves the development and implementation of a machine learning model for detecting and counting young trees using drone-captured RGB images and Digital Surface Model (DSM) data. The scripts provided here perform various steps from data preprocessing, image cropping, model training, prediction, and evaluation of the results.

## Project Structure

- `RGB and DSM Data`: Drone-captured data, including both RGB imagery and DSM files.
- `Scripts`: Python scripts to process data, train machine learning models, and predict tree locations.
- `Models`: Pre-trained and trained models for tree detection.
- `Outputs`: Directory for storing prediction results and processed files.

## Data Requirements

- **RGB Tiff File**: High-resolution RGB imagery from drone data.
- **DSM Tiff File**: Digital Surface Model data, providing elevation information.
- **Output Directories**: Directories for storing processed images and results.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Wittedewit/Peak_maximum.git
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Steps for Running the Model

### 1. Data Preparation

1. **Download Data**: Ensure you have the right RGB and DSM Tiff files.
2. **Set File Paths**: Place the paths to these files in the appropriate directories.

### 2. Data Preprocessing

- **Projection and Cropping**: Make sure both files have the same projection and are cropped to the same extent for consistent analysis.
- **Load and Crop DSM**: Load the DSM file and crop it to the relevant extent.
- **Assign Necessary Values**: Define values such as crop size.
- **Detect Peak Maxima**: Use the DSM data to find peak coordinates.

### 3. Image Processing

- **Load RGB Data**: Load the RGB Tiff and crop it to the same extent as the DSM.
- **Crop Images**: Use the detected coordinates to crop the RGB images into smaller sections.

### 4. Model Training

- **Train the Model**: Train the machine learning model if necessary, using the prepared data.
- **Load Model**: Load a pre-trained or previously trained model for prediction.

### 5. Prediction and Evaluation

- **Predict Tree Locations**: Use the model to predict tree locations.
- **Extract Coordinates**: Compile the predictions into a list with image names and coordinates.
- **Export Results**: Export the results for visualization and further analysis in GIS software like ArcGIS Pro.

## Key Functions and Their Roles

- **crop_dsm_to_rgb_extent**: Crops the DSM image to match the spatial extent of the RGB image.
- **reproject_rgb_to_dsm**: Reprojects the RGB image to match the CRS and resolution of the DSM.
- **load_DSM**: Loads and normalizes the DSM data.
- **rescale_to_255**: Rescales image data to the 0-255 range.
- **load_RGB**: Loads and crops the RGB image.
- **make_kernel**: Creates a convolution kernel for peak detection.
- **crop_around_point**: Crops images around a given point to a specified size.
- **process_images**: Processes and saves cropped images based on detected coordinates.
- **create_model**: Constructs and compiles the machine learning model.
- **predict_folder**: Predicts the class of images in a folder, focusing on those with 'RGB' in their filenames.

![Local_maxima_slangenburg](https://github.com/Wittedewit/Peak_maximum/assets/105919559/0901dd3c-0ce8-4fa8-9292-be3df0b9d899)


## Usage Example

1. **Preprocess Data**:
   ```python
   rgb_img = load_RGB(rgb_input, low_crop_y, high_crop_y, low_crop_x, high_crop_x, rgb_output_dir)
   ```

2. **Train the Model**:
   ```python
   history = model.fit(augmented_train_dataset, epochs=100, validation_data=normalized_val_dataset)
   ```

3. **Predict and Save Results**:
   ```python
   predictions = predict_folder(folder_path)
   ```

## Notes

- **Data Quality**: The quality of the input data significantly impacts model performance. Ensure high-resolution, accurately aligned images.
- **Training Data**: Increasing the variety and quantity of training data can enhance model accuracy.
- **Processing Time**: The functions, particularly those involving large datasets, may require significant computational resources.

## Future Work

- **Improve Model Accuracy**: Increase the training dataset diversity and explore more advanced models.
- **Optimize Processing**: Reduce computation times for functions handling large image datasets.
- **Expand Use Cases**: Adapt the model for other forestry management tasks like detecting diseased or dead trees.

For any issues or contributions, please refer to the GitHub repository [here](https://github.com/Wittedewit/Peak_maximum.git).
![image](https://github.com/Wittedewit/Peak_maximum/assets/105919559/a60a2a31-6339-4fc5-8d77-25c266ee8a8a)

