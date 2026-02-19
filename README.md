# Predictive Maintenance: Anomaly Detection 🔍⚙️

This repository contains a reference implementation for detecting equipment anomalies as part of the **Predictive Maintenance (PdM) framework**. 

When run-to-failure data or explicit failure labels are scarce, traditional supervised learning falls short. This project solves that by utilizing an **Autoencoder Neural Network** to perform unsupervised/semi-supervised anomaly detection based purely on normal operational sensor data.

## 🧠 Approach: The Autoencoder Method

An Autoencoder is a type of neural network designed to compress (encode) input data into a lower-dimensional space and then reconstruct (decode) it back to its original form. 

**How it detects anomalies:**
1. **Training:** The model is trained **only** on "Normal" operational data (where `target == 0`). It learns to perfectly reconstruct these healthy sensor patterns.
2. **Inference:** We pass unseen test data through the model. 
3. **Detection:** If the equipment is degrading or acting abnormally, the model will struggle to reconstruct the unfamiliar data. This results in a high **Reconstruction Error (Mean Squared Error)**. If the error exceeds our dynamic 99th-percentile threshold, the instance is flagged as an **Anomaly**.



## 📊 Dataset Requirements

This implementation is designed to work with time-series sensor data provided in `.parquet` format. 

**Expected Structure:**
- `train.parquet`: Used for scaling and training. Must contain normal operations.
- `test.parquet`: Used for inference and threshold testing.
- **Features:** A time index (`Date`), multiple continuous sensor readings (`X1`, `X2`, `X3`, `X4`, `X5`), and a binary `target` (0 = Normal, 1 = Anomaly).

## 🚀 Getting Started

### Prerequisites
To run the Jupyter/Colab Notebook, you need the following libraries:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn pyarrowM
```
Usage
1. Clone this repository.
2. Ensure your train.parquet and test.parquet files are in the same directory as the notebook.
3. Open Anomaly_Detection_Autoencoder.ipynb in Google Colab or Jupyter Notebook.
4. Run the cells sequentially to preprocess the data, build the Autoencoder, train the model, and visualize the detected anomalies.

## 📈 Results & Visualization
The notebook outputs a detailed time-series visualization. The continuous blue line represents the model's reconstruction error over time, while the dashed red line represents the automatically calculated anomaly threshold. Any data point exceeding this threshold is distinctly highlighted as a detected anomaly, allowing maintenance teams to investigate before critical failure occurs.

_Developed as a reference implementation for Predictive Maintenance frameworks_
