An AI-powered web application that classifies fruits and vegetables as healthy or rotten using deep learning, and pairs the diagnosis with live weather data to recommend which crops are currently in season.

**📋 Table of Contents**
Overview
Key Features
Technology Stack
System Architecture
Machine Learning Details
Installation Guide
Usage Instructions
Project Structure
Results & Performance
Future Improvements
Author / Credits
**Overview**

Global agricultural output has quadrupled since 1961, but manual quality inspection hasn't scaled with it. Traditional harvest inspection is slow, labor-intensive, and highly dependent on an inspector's individual experience — leaving plenty of room for costly human error.

CropCare solves this by putting a trained computer vision model behind a simple web interface. Farmers or agricultural workers upload a photo of a fruit or vegetable, and the system tells them — in seconds — whether it's healthy or rotten, along with a confidence score. It also pulls real-time weather data to suggest which crops are best suited to current conditions.

The goal is to give both smallholder farmers and larger agricultural operations a fast, low-cost, accessible tool for quality control — reducing food waste, cutting labor costs, and improving crop management decisions, without requiring any specialized technical knowledge to operate.

**Real-world applications:**

On-farm produce quality screening before distribution
Reducing food waste through early detection of spoilage
Supporting data-driven crop planning using weather-informed suggestions
Providing an accessible tool for regions with limited access to agricultural extension services
**Key Features**
🖼️ Image-Based Classification — Upload a photo and get an instant healthy/rotten classification
🎯 Confidence Scoring — Every prediction is returned with a confidence percentage and top alternative predictions
🌦️ Weather-Aware Recommendations — Integrates live weather data to suggest in-season crops
⚡ Real-Time Processing — Results delivered in seconds, suitable for field use
📱 Responsive Web Interface — Works across desktop, tablet, and mobile devices
🗂️ Result History & Storage — Classification results, images, and metadata are stored for future reference
🔌 API-First Design — RESTful backend endpoints allow integration with external agricultural management systems
🧩 Modular Architecture — Clean separation between data processing, ML inference, and UI layers for easy extension
🛡️ Graceful Error Handling — Clear feedback for unsupported formats or low-quality uploads

**Technology Stack**
**Category	**                     ** Technologies**
Frontend	                        React.js
Backend	                          Flask / Django (Python)
Machine Learning	                TensorFlow, PyTorch, CNN (Convolutional Neural Network)
Pretrained Models	                MobileNetV2, ResNet50 (Transfer Learning)
Image Processing	                OpenCV
Database	                        MySQL (structured data & logs), MongoDB (image metadata)
External APIs	                    OpenWeatherMap API
Model Training Environment	      Google Colab (GPU/TPU-accelerated)
Deployment	                      AWS, Google Cloud Platform
Model Optimization	              TensorFlow Lite / ONNX (edge inference)
Testing	                          Selenium (UI), Postman (API), Jupyter Notebooks (model validation)
Version Control	                  Git & GitHub
IDE	                              Visual Studio Code

User uploads image
        │
        ▼
┌───────────────────┐
│  User Interface    │  React web app — image upload, results display, weather dashboard
└─────────┬──────────┘
          │ HTTP Request
          ▼
┌───────────────────┐
│  Application Layer │  Flask/Django backend — API endpoints, auth, business logic
└─────────┬──────────┘
          │
          ▼
┌───────────────────┐
│  Service Layer     │  Image preprocessing — filtering, resizing, augmentation
└─────────┬──────────┘
          │
          ▼
┌───────────────────┐
│  ML Inference Layer│  CNN model (transfer learning) — binary classification
└─────────┬──────────┘
          │
          ▼
┌───────────────────┐         ┌──────────────────┐
│  Weather Layer      │◄──────┤  OpenWeatherMap API │
└─────────┬──────────┘         └──────────────────┘
          │
          ▼
┌───────────────────┐
│  Data Layer         │  MySQL / MongoDB — image storage, results, user history
└─────────┬──────────┘
          │
          ▼
   Results + Suggestions
      returned to user


User uploads an image via the web interface.
Backend validates and stores the image.
Image is preprocessed (resized to 224×224, normalized, augmented if needed).
The CNN model runs inference and returns a classification with a confidence score.
The backend fetches current weather data in parallel.
Classification results and weather-based crop suggestions are combined and stored.
Results are returned to the user through the web dashboard.
Machine Learning Details
Dataset
Labeled image dataset covering multiple crop types, each split into Healthy and Rotten classes (e.g., Apple, Banana, Bell Pepper, Carrot, Cucumber, Orange, Potato, Tomato).

**Preprocessing Pipeline**
Filtering — noise and artifact removal to improve image quality
Resizing & Normalization — standardized to 224×224 pixels, pixel values scaled to [0, 1]
Data Augmentation — rotation, flipping, and zooming applied to increase dataset diversity and reduce overfitting
**Model Architecture**
Base Model: Transfer learning using pretrained MobileNetV2 or ResNet50
Custom Layers: Additional dense layers with dropout for fine-tuning on the crop dataset
Output Layer: Sigmoid activation for binary classification (healthy vs. rotten)
**Training Approach**
Fine-tuning of upper layers on top of frozen pretrained base layers
Hyperparameter tuning (learning rate, batch size) using a validation set
Optimization techniques: early stopping, learning rate decay, and model checkpointing to prevent overfitting
**Evaluation Metrics**
Accuracy, Precision, Recall, F1-score, and confusion matrix analysis were used to guide model refinement.
**Target Performance**
Target classification accuracy: ≥ 95%
Target inference/response time: a few seconds per image, suitable for real-time field use
<img width="1200" height="400" alt="training_history" src="https://github.com/user-attachments/assets/6c520876-a2ff-45be-a4c6-0403cc9af2d1" />


**System Architecture**

CropCare follows a layered architecture, taking an uploaded image through preprocessing, inference, and enrichment before returning a result to the user.
