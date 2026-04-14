Facial Paralysis Detection and Recovery Tracking Using Deep Learning and Digital Twin Models

Project Guide : 
Dr. Pooja Agarwal
Professor, Department of Computer Science and Engineering 
PES University 

Batch 201 :
- Abhishek P
- Bhuvi Prashanth
- Harsha
- Shreya Parashar
  
Project Overview
Facial paralysis diagnosis currently relies on subjective visual assessment via the House-Brackmann (HB) Scale. This project transitions from subjective visual inspection to objective measurement by utilizing a modular AI pipeline that fuses high-precision facial geometry with deep texture analysis .The system provides a comprehensive framework for automated detection, clinical severity grading, and longitudinal recovery tracking through a personalized Digital Twin engine .

Implementation Status (10% Progress)
The repository currently contains the Foundation Modules of the system :
- Geometric Landmark Engine (Module 3): Successfully implemented a 468-point 3D landmark extraction pipeline using MediaPipe FaceMesh .
- FAI Computation Engine: Fully functional real-time logic for the Facial Asymmetry Index (FAI) based on Euclidean distance deviations across 5 standardized facial tasks .
- Baseline Generation: Autonomous generation of a Personalized Facial Baseline for recovery benchmarking .
- Performance Metrics: Confirmed stable 30+ FPS landmark tracking on standard mobile device CPUs 

Technical StackIntelligence & ModelsMediaPipe FaceMesh: 
* Real-time 468-point 3D landmarking for geometric symmetry analysis.
* ResNet50 (PyTorch): Pretrained backbone utilized via Transfer Learning for 2048-dim deep texture feature embeddings.
* Severity Classifier: Custom ML models employing Ordinal Regression and Multilayer Perceptron (MLP) architectures.
* Explainable AI (XAI): Integration of LIME (Local Interpretable Model-agnostic Explanations) and Grad-CAM for diagnostic transparency 

APIs & ToolsLanguages: 
* Python 3.x.
* Processing: OpenCV (ROI extraction, alignment), NumPy, and Pandas (Longitudinal data handling) .
* Frontend: Streamlit-based clinical diagnostic pipeline.

Repository Structure

├── app/
│   ├── app.py              # Clinical Streamlit interface for diagnostic pipeline
│   └── pipeline.py         # Modular pipeline logic (Preprocessing -> Feature Extraction)
├── data_tools/
│   ├── extract_palsynet.py # Dataset downloader and frame extraction utility
│   └── labeling.py        # Asymmetry-based auto-labeling script (dlib/MediaPipe)
└── metadata/               # Extraction logs, dataset info, and patient history records

Methodology: The Modular Pipeline
The system is architected as a 7-module pipeline to ensure edge-deployment stability and clinical interpretability :
1. Data Acquisition: Capture of task-based facial videos (Neutral, Smile, Eye Closure, Brow Raise) .
2. Preprocessing: Real-time face detection, alignment, and coordinate normalization .
3. Feature Extraction: 3D landmark mapping and FAI computation .
4. Feature Fusion: Merging geometric landmark vectors with ResNet50 texture embeddings .
5. Severity Classification: Ordinal Regression mapping features to House-Brackmann Grades .
6. Digital Twin Engine: Longitudinal modeling for recovery trajectory forecasting .
7. Explainable AI: Generation of importance maps and automated clinical reports .

Dataset Strategy
The system utilizes a curated hybrid dataset of approximately 9,300 frames:
- AFLFP: 5,632 images (88 subjects) for base geometric symmetry training.
- MEEI: 480 images and 60 videos for clinical paralysis benchmarks.
