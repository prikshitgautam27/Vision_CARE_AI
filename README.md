# VISIONCARE AI
( IDEA Credits : Prikshit Sharma)
(Reports Credits: Kartik Bansal)

**Tagline:** Early Detection, Clearer Tomorrow

---

## Overview

**VISIONCARE AI** is an advanced cataract detection platform leveraging ensemble deep learning models to assist clinics in delivering rapid, reliable diagnostics. Built with **Streamlit** and **Keras**, the app generates automated diagnostic reports tailored for clinic workflows and supports a public-facing dashboard for transparency.

---

## Features

- **Robust Preprocessing Pipeline:**  
  Automatically converts input images to grayscale, applies vignette, resizes, and normalizes them for enhanced prediction consistency.
- **Ensemble Deep Learning:**  
  Utilizes two powerful CNN base models with a meta-model for improved detection accuracy.
- **Interactive Streamlit Interface:**  
  Patient info input, image uploads, and results—all in a user-friendly UI.
- **Styled Diagnostic Report:**  
  Probability output from meta-model and final classification, formatted in a clinic-friendly layout.
- **Downloadable Reports:**  
  Save and share diagnostic results as reports for each patient.
- **Public Dashboard:**  
  View all processed reports in one place, enabling clinics to track exams and outcomes.
- **Business Model Integration:**  
  Pricing details, membership options, and clinic registration built into the app.
- **Developer Credits:**  
  Developer details with clickable LinkedIn profiles.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prikshitgautam27/Vision_CARE_AI.git
   cd Vision_CARE_AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model files: or Run Stacking-approach to obtain model files**
   Place the following files in the project directory:
   - `base_model_1.h5`
   - `base_model_2.h5`
   - `meta_model.keras`

---
4.**Add Kaggle DAtaset to working Folder**
 - `https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k`
## Usage

1. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

2. **App Workflow:**
   - Upload a fundus image.
   - Enter patient details.
   - View and download the diagnostic report.
   - Newly generated reports automatically appear on the public dashboard.

---

## Screenshots / Demo

| Original Image | Preprocessed Image | Styled Report | Dashboard |
|:--------------:|:-----------------:|:-------------:|:---------:|
| ![Original Image](path/to/original_screenshot.png) | ![Preprocessed Image](path/to/preprocessed_screenshot.png) | ![Report Screenshot](path/to/report_screenshot.png) | ![Dashboard](path/to/dashboard_screenshot.png) |

*Replace these with actual screenshots.*

---

## Business Model

- **Membership:** Yearly discounts for frequent users.
- **Clinic Registration:** Clinics can register via the app for bulk processing and dashboard monitoring.

**For business inquiries:**  
📧 pgautamlinked@gmail.com

---

## Developer Info

Developed by **Prikshit Gautam & Team**  
[Prikshit Gautam - LinkedIn](https://www.linkedin.com/in/prikshitgautam27/)

---

## Future Work / Roadmap

- Add Grad‑CAM explainability overlays for visual model interpretation.
- Enable PDF report download.
- Integrate with cloud storage for global access.
- Deploy on Streamlit Cloud or Hugging Face Spaces.

---

## License

This project is licensed under the [MIT License](LICENSE).
