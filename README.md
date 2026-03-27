# 👕 AI-Powered Visual Product Search Engine

An end-to-end Machine Learning pipeline that takes an image of a clothing item and instantly retrieves the top 5 visually similar products from a massive dataset. Built with a custom **ResNet50** architecture, **FAISS** vector search, and a **Streamlit** web interface.

## 🌟 Project Highlights
* **High-Accuracy Classification:** Achieved a **98.95% validation accuracy** in categorizing fashion items across 7 distinct classes.
* **Deep Feature Extraction:** Utilizes a truncated ResNet50 model to extract 2048-dimensional feature vectors from images, capturing complex geometric and textural patterns.
* **Lightning-Fast Retrieval:** Implements Meta's FAISS (Facebook AI Similarity Search) to perform sub-second nearest-neighbor matching across a database of 35,000+ items.
* **MLOps Integration:** Experiment tracking, hyperparameter tuning, and model metrics were systematically logged using **MLflow**.

## 🏗️ System Architecture
1. **The Classifier (`best_classifier_model.h5`):** A fine-tuned ResNet50 model that analyzes the uploaded image and predicts the core product category (e.g., Apparel, Footwear, Accessories) with high confidence.
2. **The Extractor (`resnet50_extractor.h5`):** A modified ResNet50 model with the final classification layer removed, acting purely as a "shape and texture" scanner to generate embeddings.
3. **The Index (`product_index.index`):** A pre-computed FAISS database containing the spatial vectors of the entire Myntra inventory.
4. **The Frontend (`search_engine.py`):** A responsive, dark-mode web application built in Streamlit for real-time user interaction.

## 📊 Model Performance Tracking
Training runs and metric evaluations were actively monitored using MLflow.
* **Optimizer:** Adam (Learning Rate: 0.001)
* **Batch Size:** 64
* **Accuracy:** 98.95%
* **Precision:** 98.83%
* **Recall:** 98.95%
* **F1-Score:** 98.85%

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow / Keras (ResNet50)
* **Computer Vision:** OpenCV / Pillow
* **Vector Database:** FAISS (L2 Distance Normalization)
* **Web Framework:** Streamlit
* **MLOps:** MLflow

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/gokz7/visual-product-search-system.git](https://github.com/gokz7/visual-product-search-system.git)
