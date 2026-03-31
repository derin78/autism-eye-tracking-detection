# Autism Detection Using Eye Tracking System

A full-stack, webcam-based computer vision application designed to screen for gaze patterns associated with Autism Spectrum Disorder (ASD). 

Built with **Python**, **OpenCV**, **MediaPipe Face Mesh**, **Scikit-Learn**, and **CustomTkinter**.

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **Modern GUI** | Easy-to-use Control Panel built with CustomTkinter (dark theme). |
| **Camera Selection** | Dropdown to instantly switch between laptop webcams or connected phone cameras (e.g., Pixel via USB) for higher resolution tracking. |
| **Real-Time Eye Tracking** | Uses MediaPipe (468 facial landmarks) to extract sub-pixel iris coordinates without expensive infrared hardware. |
| **Autism Screening Test** | Automated 10-image test based on the clinical **Preferential Looking Paradigm** (face vs. object visual attention). |
| **Machine Learning Classifier** | Random Forest model trained on synthetic data derived from published clinical thresholds predicting ASD risk scores. |
| **Automated Reports** | Generates printable text reports saved under the subject's name with classification, risk score, and interpretation guide. |
| **Gaze Analytics** | Extracts fixations, saccades, and scanpath features, and generates 2D visual heatmaps of the user's gaze. |

---

## 🚀 Quick Start

### 1. Install dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Start the graphical user interface (GUI):
```bash
python main.py
```

### 3. How to Run a Screening Test
1. Launch the app and select your camera from the **📷 Camera** dropdown.
2. Click **▶ Start Tracking** to activate the webcam and eye-tracking engine.
3. Click **🎯 Calibrate** (Optional but recommended) and follow the 9 dots on the screen.
4. Click **🧪 Run Screening Test**, enter the subject's name, and have them look naturally at the 10 fullscreen images.
5. View the results instantly in the **📋 Results** panel and find the detailed text report saved in the `experiments/reports/` folder.

---

## 🧠 How the Science Works

### The Screening Method (Preferential Looking Paradigm)
The system displays split-screen stimulus images featuring a social element (faces) on one side and a non-social element (objects/shapes) on the other. 
* **Typical Development:** Research shows neurotypical individuals naturally gravitate toward social information (looking at faces >55% of the time).
* **ASD Risk Pattern:** Individuals on the Autism Spectrum often show reduced social visual attention (looking at faces <45% of the time).

### The Machine Learning Model
The system uses a **Random Forest Classifier** to assess the gaze data. 
Because real clinical patient logs are restricted by medical privacy laws, the model is trained on **400 synthetic samples**. Using `numpy`, we mathematically generated training data strictly constrained to the clinical thresholds published in ASD eye-tracking medical journals. 

*Run the model evaluation script to see the exact metrics (94.75% CV Accuracy):*
```bash
python evaluate_model.py
```

---

## 📁 Project Structure

```
AutismEyeProject/
├── main.py                  # App entry point (Launches GUI)
├── config.py                # Central configuration (thresholds, resolutions)
├── evaluate_model.py        # ML evaluation script (Accuracy, AUC-ROC, etc.)
├── requirements.txt
│
├── app/
│   ├── gui.py               # CustomTkinter User Interface
│   ├── test_session.py      # Runs the 10-image screening test
│   ├── classifier.py        # Random Forest ML Model & Synthetic Data generator
│   ├── report.py            # Text report generator
│   ├── heatmap.py           # Gaussian heatmap generator
│   └── visualizer.py        # Gaze overlay visualizations
│
├── tracking/
│   ├── tracking.py          # Core GazeTracker (MediaPipe Iris, EMA smoothing)
│   ├── calibration.py       # 9-point fullscreen calibration routine
│   └── gaze_mapper.py       # Affine iris→screen mapping math
│
├── stimuli/
│   └── generate_stimuli.py  # Script that programmatically drew the test images
│   └── images/              # The 10 face vs object test images
│
├── data/
│   ├── processsing.py       # Cleans and segments raw gaze data
│   ├── features.py          # Extracts fixations, saccades, and scanpath metrics
│   └── raw/, processed/, features/
│
├── experiments/             # Output folder for Heatmaps, Reports, & ML eval
└── docs/
    ├── README.md            # This file
    └── Project_Cheat_Sheet.pdf # Simple explainer for presentations
```

---

## ⚠️ Disclaimer
**This software is a proof-of-concept research and screening tool.** It utilizes a standard webcam and synthetic training data. It does **not** provide a medical diagnosis of Autism Spectrum Disorder. Any clinical concerns should be directed to a qualified medical professional. 
