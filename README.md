# Breast-Cancer-Stage-Prediction
In this work, we developed a multimodal 2.5D CNN to predict the T-stage of breast tumors using clinical, radiomic, and imaging features. Multimodal approaches are becoming increasingly prevalent in medical-imaging machine-learning pipelines, and this work combines structured clinical data with 3D MRI volumes from multiple modalities to improve predictive performance. 

**Clone the repository**
```bash
git clone https://github.com/grod23/Breast-Cancer-Stage-Prediction.git
cd Breast-Cancer-Stage-Prediction
```

### Create Virtual enviroment
```python -m venv venv
source venv/bin/activate       # macOS/Linux
or
venv\Scripts\activate          # Windows
```
### Install Required Packages
```
pip install -r requirements.txt
```
### Run Script
```
python main.py
```
### Breast MRI scans preprocessing
Utilizing the MONAI framework, MRI volumes go through the following transformations:
- Cropping full volume to 3 slice ROI (Region of Interest)
- Standardization of voxel spacing to (1.0, 1.0, 1.0)
- Resizing height and width to (360, 360)
- Z-Score standardization
- Min-Max Normalization

## 2.5D CNN
The 2.5D CNN serves as a comrpomise between 2D and 3D convolutional neural networks. In medical imaging, where CT and MRI scans are inherently 3-Dimensional, this balance is crucial. The core idea is that we stack a small amount of neighboring slices (3-7) along the channel dimension. This strategy transforms 3D structural information while retaining a 2D image's spatial shape. This approach allows us to train a limited 3D volume with a 2D CNN. Therefore we gain the beneficial speed and resolution of a 2D CNN paired with the volumetric context of a 3D CNN. 

### Grad-CAM Visualization
XAI (Explainable AI) is essential for interpreting model decisions and gaining the trust of workers in their respective industry, especially in the field of medical imaging. The following is a sample GradCAM (Gradient Class Activation Map) visual that highlights image regions most influential to the model's prediction.

### Classification Report
                   precision    recall   f1-score    support

       T-Stage 1       0.75      0.54      0.62        84
       T-Stage 2       0.62      0.65      0.64        75
       T-Stage 3       0.63      0.87      0.73        52
       T-Stage 4       0.94      1.00      0.97        17

        accuracy                           0.68        228
       macro avg       0.74      0.76      0.74        228
    weighted avg       0.70      0.68      0.68        228

### Confusion Matrix
The confusion matrix assists in understanding where the model is confusing varying T-stage values. 


![Confusion Matrix](Results/confusion_matrix.png)


### Future Works
Our work is just the stepping stone to exploring machine learning models capable of predicting the full TNM staging. To advance our models practicality, we plan on incorporating the prediction of all 3 Tumor, Nodal, and Metastasis labels. Furthermore, we could translate these labels into a tumor stage prognosis ranging from stage I-IV. 
 
