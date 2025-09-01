# 📈 Time Series Anomaly Detection  

This repository explores **detecting anomalies in time series data** — sudden spikes, drops, or shifts that break expected patterns. The project combines careful **data preprocessing**, **artificial dataset generation**, and a **GRU Autoencoder** model to evaluate anomaly detection performance.  

---

## 🎯 Project Goals  
- Create **artificial test datasets** with controlled anomalies and shifts.  
- Develop preprocessing pipelines: alignment, deseasonalization, differencing, scaling.  
- Train a model to **reconstruct normal patterns** and detect deviations.  
- Compare anomaly detection methods using metrics like **PATE**.  
- Reflect on what works (and what doesn’t) for robust anomaly detection.  

---

## 🖼️ Project Poster(open as pptx for the better quality)

<p align="center">
  <img src="https://github.com/pavlesav/Time-Series-Anomaly-Detection/blob/main/poster.png" alt="Project Poster" width="100%">
</p>



---

## 🔧 Workflow  

### 1. Dataset Creation  
- Python function to generate **5 artificial test datasets**.  
- Inserted **group anomalies** of different lengths.  
- Applied **dataset shifts** to misalign training and test sets.  
- Stored **true labels and shift info** for evaluation.  

### 2. Preprocessing  
- **Alignment:** used `scipy.correlate` to estimate lag and shift datasets.  
- **Deseasonalization:** disjoint window averaging.  
- **Detrending:** first-order differencing  
  \[
  ∇_1[x](t) = x_t - x_{t-1}
  \]  
- **Scaling & Normalization:** robust scaling + log transform.  

### 3. Modeling & Detection  
- **GRU Autoencoder** with 4 layers `[64, 32, 32, 64]`.  
- Regularization: dropout, batch normalization, learning rate reduction.  
- Trained on window-based sequences of normalized input.  
- **Anomaly score:** mean absolute reconstruction error.  
- Postprocessing: dataset reshift, threshold testing, visualization.  

### 4. Evaluation  
- **Synthetic experiments** across thousands of datasets.  
- **PATE metric** for anomaly detection performance.  
- Visualization of preprocessing, anomaly predictions, and reconstruction errors.  
- Insights:  
  - Alignment successful (average error: 2.37 over 5000 datasets).  
  - Threshold tuning didn’t improve PATE significantly.  
  - Preprocessing is critical — deseasoning, scaling, and alignment strongly influence results.  

---

## 📊 Key Takeaways  
- Preprocessing is as important as the model.  
- Visualizations help interpret anomaly patterns.  
- Baselines are tough to beat — thresholds alone don’t add much.  
- Stationarity and alignment make or break anomaly detection.  

---

Project for *Knowledge Discovery and Data Mining 2*.  


