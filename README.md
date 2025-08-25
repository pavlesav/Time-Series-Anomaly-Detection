# 📈 Time Series Anomaly Detection 

This project focuses on **detecting anomalies in time series data** using a **GRU (Gated Recurrent Unit) Autoencoder**.  
The work was developed as part of the **Knowledge Discovery and Data Mining 2** course, in collaboration with Konstantin Benischke and Pavle Savković.

---

## 🔹 Project Overview
Time series often contain unexpected spikes and shifts. The goal of this project is to:
1. Generate artificial test datasets with controlled anomalies.
2. Align and preprocess time series data (deseasonalization, differencing, scaling).
3. Train a **GRU Autoencoder** to reconstruct normal patterns.
4. Detect anomalies based on **reconstruction error**.
5. Evaluate detection quality using metrics like **PATE**.

## 🛠️ Methods & Techniques

### 🔧 Preprocessing
- Dataset alignment using `scipy.correlate` for lag estimation.
- Deseasonalization with disjoint window averaging.
- First-order differencing:  
  \[
  ∇_1[x](t) = x_t - x_{t-1}
  \]
- Robust scaling & log normalization.

### 🤖 Model: GRU Autoencoder
- **Architecture:** 4 GRU layers `[64, 32, 32, 64]`.
- **Regularization:** dropout, batch normalization, learning rate reduction.
- **Training:** sequence windows, log-normalized inputs.
- **Anomaly detection:** mean absolute reconstruction error.

### 📊 Evaluation
- Artificial datasets with injected anomalies of varying lengths.
- Realignment of shifted test sets.
- Metrics: anomaly score distributions, reconstruction error, PATE score.

---