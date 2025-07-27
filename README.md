# Vocal Stress Analysis for Deception Detection 🎙️

A proof-of-concept project that uses machine learning to analyze vocal biomarkers in speech and classify the speaker's emotional stress level, serving as an investigative aid for deception detection.

## ## Core Features
* **Vocal Feature Extraction**: Extracts key prosodic and spectral features from audio files, including Pitch ($F_0$), Jitter, Shimmer, and MFCCs.
* **Data Processing Pipeline**: Automatically labels data from filenames, scales features for model compatibility, and prepares the dataset for training.
* **Baseline Model**: Implements a Logistic Regression classifier to establish a baseline performance for stress detection.
* **Model Evaluation**: Generates an accuracy score, a detailed classification report, and a confusion matrix to evaluate model performance.

---
