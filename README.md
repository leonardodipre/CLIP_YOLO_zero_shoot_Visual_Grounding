Here is an enhanced README with detailed information about the main approach, including the loss calculation and IoU for bounding boxes in visual grounding:

---

# CLIP_YOLO_zero_shoot_Visual_Grounding

## Introduction

This project combines CLIP and YOLO for zero-shot visual grounding. By leveraging the strengths of both models, the aim is to accurately identify and localize objects in images based on natural language descriptions. The approach achieves a 30% accuracy on the RefCOCOg dataset.

## Repository Structure

- `Dataloader.py`: Handles data loading and preprocessing.
- `main.py`: Main script for running the visual grounding experiments.
- `utils_.py`: Utility functions used across the project.
- `yolov5s.pt`: Pre-trained YOLOv5s model weights.
- `__pycache__`: Cached files.

## Approach

1. **Data Preparation**: Images, bounding boxes, and sentences are taken from the RefCOCOg dataset.
2. **Model Integration**: CLIP is used for natural language understanding and feature extraction, while YOLO is used for object detection.
3. **Zero-Shot Grounding**: The combined model processes the input text and image to identify and localize the referred object without prior training on the specific dataset.

## Code Overview

### main.py

The `main.py` script orchestrates the visual grounding process. Here’s a step-by-step breakdown:

1. **Imports and Initial Setup**:
   - Import necessary libraries including `torch`, `CLIP`, `YOLO`, and utility functions.
   - Set up device configuration (CPU/GPU).

2. **Load Models**:
   - Load the pre-trained CLIP model for extracting features from both images and text.
   - Load the pre-trained YOLO model for detecting objects in images.

3. **Data Loading**:
   - Utilize `Dataloader.py` to load and preprocess the RefCOCOg dataset.
   - Prepare images, bounding boxes, and corresponding sentences.

4. **Feature Extraction**:
   - Extract visual features from images using the YOLO model.
   - Extract textual features from sentences using the CLIP model.

5. **Matching and Grounding**:
   - Compare the extracted features to find the best match between the textual description and the detected objects.
   - Calculate similarity scores between the textual and visual features.
   - Identify and localize the referred object in the image based on the highest similarity score.

6. **Loss Calculation**:
   - Use Intersection over Union (IoU) to measure the overlap between predicted and ground truth bounding boxes.
   - Calculate the loss as \( \text{Loss} = 1 - \text{IoU} \), where IoU is the ratio of the intersection area to the union area of the predicted and ground truth boxes.

7. **Evaluation**:
   - Evaluate the performance of the model by comparing the predicted bounding boxes with the ground truth.
   - Calculate accuracy and other relevant metrics.

8. **Results**:
   - Print and save the results, including the accuracy achieved on the RefCOCOg dataset.


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/leonardodipre/CLIP_YOLO_zero_shoot_Visual_Grounding.git
   cd CLIP_YOLO_zero_shoot_Visual_Grounding
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt TODO
   ```

3. Ensure you have the necessary pre-trained models and datasets.

## Usage

Run the main script to start the visual grounding process:
```bash
python main.py
```

## Results

The current model achieves a 30% accuracy on the RefCOCOg dataset. Further improvements and optimizations are ongoing.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

---

For more details, visit the [repository](https://github.com/leonardodipre/CLIP_YOLO_zero_shoot_Visual_Grounding).