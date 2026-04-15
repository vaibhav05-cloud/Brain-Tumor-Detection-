# Brain Tumor Detection and Classification

## Overview
This project presents an intelligent system for brain tumor detection and classification using deep learning techniques. The system analyzes MRI scan images to determine the presence of a tumor and classify it into different categories.

The primary objective of this project is to assist medical professionals by providing a fast, reliable, and automated diagnostic tool. Early detection of brain tumors is crucial, as tumors can vary significantly in size, shape, and location, making manual diagnosis complex and time-consuming.

---

## Live Application
The project is deployed and accessible at:
https://brain-tumor-detection-2-fyay.onrender.com/

---

## Features
- Automated detection of brain tumors from MRI images  
- Multi-class classification (Glioma, Meningioma, Pituitary, No Tumor)  
- User-friendly web interface  
- Real-time prediction results  
- Scalable deployment using Flask  

---

## Problem Statement
Brain tumor diagnosis using MRI images presents several challenges:
- Variability in tumor size, shape, and position  
- Low contrast between tumor and surrounding tissues  
- Large volume of medical imaging data  

Manual analysis by radiologists can be time-consuming and may lead to inconsistencies. This project aims to automate the detection and classification process using deep learning techniques to improve efficiency and accuracy.

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Scikit-learn  
- Flask  

---

## Methodology

### 1. Data Collection
MRI image dataset containing both tumor and non-tumor cases.

### 2. Data Preprocessing
- Image resizing  
- Normalization  
- Noise reduction  

### 3. Model Development
A Convolutional Neural Network (CNN) is used for feature extraction and classification. CNNs are effective for image-based tasks due to their ability to learn spatial features automatically.

### 4. Training and Evaluation
- Dataset split into training and testing sets  
- Model trained on labeled MRI images  
- Performance evaluated using accuracy and loss metrics  

### 5. Deployment
The trained model is integrated into a Flask web application for real-time predictions.

---

## System Architecture
- Input: MRI Image  
- Processing: Image Preprocessing + CNN Model  
- Output: Tumor Detection and Classification  

---

## Installation and Setup

### Prerequisites
- Python 3.x  
- pip  

### Steps
```bash
# Clone the repository
git clone <your-repository-link>

# Navigate to project directory
cd brain-tumor-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

## Usage
1. Open the web application  
2. Upload an MRI image  
3. Click on "Predict"  
4. View the result (Tumor type or No Tumor)  

---

## Results
The model achieves strong performance in detecting and classifying brain tumors from MRI scans. Deep learning models such as CNNs provide high accuracy and reliability in medical image classification tasks.

---

## Future Enhancements
- Improve accuracy using advanced architectures (ResNet, EfficientNet)  
- Add tumor segmentation for precise localization  
- Integration with hospital systems  
- Support for 3D MRI data  

---

## Conclusion
This project demonstrates the application of deep learning in healthcare, specifically in medical image analysis. By automating brain tumor detection and classification, the system helps in faster and more accurate diagnosis.

---

## License
This project is intended for educational and research purposes only.