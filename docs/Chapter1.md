# Chapter 1
# Introduction

## 1.1. Introduction
Autism Spectrum Disorder (ASD) is a lifelong developmental disability that affects how people communicate and interact with the world. Early intervention can significantly improve the quality of life and developmental outcomes for children with ASD. One of the key early indicators of ASD is a difference in visual attention, specifically a reduced tendency to look at faces and social stimuli compared to typically developing children. This behavioral pattern is often assessed clinically using the Preferential Looking Paradigm (PLP).

Traditionally, evaluating these gaze patterns requires specialized, expensive infrared eye-tracking hardware, limiting the accessibility of early screening mostly to specialized clinics. This project aims to bridge this gap by developing an accessible, low-cost desktop application that leverages standard webcams and computer vision techniques to track eye movements and provide an initial screening for ASD.

## 1.2. Problem Statement
The primary problem addressed by this project is the lack of accessible and affordable early screening tools for Autism Spectrum Disorder. Current clinical eye-tracking systems are prohibitively expensive, require specialized environments, and are not widely available in general healthcare settings or schools. As a result, many children experience delays in diagnosis and miss the crucial window for early intervention. There is a critical need for a reliable, software-based screening tool that can operate on standard consumer hardware (like laptops with webcams) to identify children who may be at high risk for ASD and should be referred for comprehensive clinical evaluation.

## 1.3. Background and Literature Review

### 1.3.1. Literature Review
Extensive research has demonstrated that children with ASD exhibit atypical visual attention patterns. Studies using the Preferential Looking Paradigm (PLP) consistently show that when presented with simultaneous social (e.g., faces) and non-social (e.g., objects or geometric shapes) stimuli, typically developing children show a strong preference for the social stimuli. In contrast, children with ASD often demonstrate reduced attention to faces and may prefer the non-social objects. 

Recent advancements in artificial intelligence and computer vision have made it possible to perform gaze estimation using standard webcams without active infrared illumination. Frameworks like MediaPipe provide robust facial landmark detection, enabling the extraction of iris coordinates. By combining these computer vision techniques with machine learning classifiers, it is now feasible to automate the analysis of gaze data and identify patterns associated with ASD.

### 1.3.2. Methodology

#### 1.3.2.1. Research Methodology
The research methodology for this project involved studying clinical literature to understand the Preferential Looking Paradigm and the specific gaze metrics (such as face attention ratio, fixation counts, and saccades) that differentiate typical from atypical development. Because real patient data is protected by medical privacy laws, we generated a synthetic dataset based strictly on the statistical thresholds and findings published in peer-reviewed medical journals. This dataset was then used to train and evaluate a Random Forest machine learning classifier to predict ASD risk levels based on the extracted gaze features.

### 1.3.3. Software Development Methodology

#### 1.3.3.1. Iterative Model
The project was developed using the Iterative Process Model. This approach allowed us to build the system incrementally. We started with a basic prototype for webcam access and face detection. In subsequent iterations, we integrated the MediaPipe iris tracking, added the calibration module, developed the screening test interface, and finally incorporated the machine learning classifier and report generation. Each iteration was tested and refined before moving on to the next, ensuring that core functionalities like real-time tracking were stable before adding complex analytical features.

### 1.3.4. Requirements

#### 1.3.4.1. Computer Programming Language (Python)
The entire application is written in Python (version 3.8+). Python was chosen due to its extensive ecosystem of data science, machine learning, and computer vision libraries, as well as its platform independence and rapid development capabilities.

#### 1.3.4.2. Libraries and Frameworks
The system relies on several key libraries:
* **OpenCV:** Used for video capture, image processing, and rendering the visual overlays (heatmaps and gaze trails).
* **MediaPipe:** Developed by Google, this framework provides the core Face Mesh model capable of tracking 468 facial landmarks, including the iris, in real time.
* **Scikit-Learn:** Used to implement the Random Forest classifier that analyzes the gaze data and predicts the risk level.
* **NumPy & Pandas:** Used for numerical computations, data manipulation, and handling the gaze data CSV files.
* **CustomTkinter:** Used to build the modern, dark-themed Graphical User Interface (GUI).

#### 1.3.4.3. Software and Tools
* **Visual Studio Code (VS Code):** Used as the primary Integrated Development Environment (IDE) for writing and debugging the code.
* **Git & GitHub:** Used for version control and source code management.
* **PyInstaller:** Used to package the final application into a standalone executable file for easy distribution.

## 1.4. The Aims of the Project
The main objectives of this project are:
1. To develop a real-time eye-tracking system that works with standard consumer webcams without requiring specialized hardware.
2. To implement a computerized version of the Preferential Looking Paradigm for ASD screening.
3. To train a machine learning model capable of classifying gaze patterns into typical and high-risk categories based on established clinical research.
4. To design an intuitive and user-friendly desktop application that can be easily operated by teachers, parents, or general practitioners.
5. To automatically generate detailed, printable reports summarizing the screening results and gaze statistics.

## 1.5. Project Limitations
While the system provides a valuable proof-of-concept for accessible screening, it has some limitations:
1. **Hardware Constraints:** Because it relies on a standard webcam rather than a dedicated infrared eye tracker, the accuracy is dependent on good lighting conditions and camera resolution.
2. **Head Movement:** The user must keep their head relatively still during the test. While the system compensates for minor head movements, significant shifts can disrupt the iris tracking.
3. **Synthetic Data:** Due to strict medical privacy laws regarding patient data, the machine learning classifier was trained on synthetic data derived from published clinical thresholds rather than real patient recordings. 
4. **Not a Diagnostic Tool:** This software is strictly a screening tool for research purposes. It cannot and does not provide a formal medical diagnosis of Autism Spectrum Disorder.

## 1.6. Report Layout
The remainder of this report is organized as follows:
* **Chapter 2 (Design and Implementation):** Details the system architecture, UML diagrams (Flow Chart, Use Case, Activity, Sequence), and the implementation of the user interface and core tracking modules. It also covers the evaluation of the machine learning model and system testing.
* **Chapter 3 (Conclusions and Future Directions):** Summarizes the project outcomes, discusses the results, and outlines potential improvements and future work for the system.
