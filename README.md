[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)
# IIT-Madras-DA2401-Machine-Learning-Lab-End-Semester-Project

## 📌 Purpose of this Template

This repository contains the complete End-Semester Project submission, including all from-scratch code, data, and the final report.
> **Scope (as per assignment brief):**
> This repository contains a complete from-scratch Python implementation of a multi-class classification system to predict digits (0-9) from the MNIST dataset.

My final submitted model (main.py) is a Tuned K-Nearest Neighbors (k=5) Classifier combined with a Principal Component Analysis (PCA) pre-processing step.

This model was chosen after a rigorous experimental process, detailed in the End-Semester Project Report_ MNIST Digit Classifier.pdf. I found that my tuned KNN model (F1 Score: 0.9568) outperformed all other models I built, including complex stacked ensembles (F1 Score: 0.9489), proving to be the most accurate, robust, and fastest solution.

---

**Important Note:** 
1. TAs will evaluate using the `.py` file only.
2. All your reports, plots, visualizations, etc pertaining to your solution should be uploaded to this GitHub repository

---

## 📁 Repository Structure

* main.py: The main executable file to train and evaluate our final model (KNN).

* algorithms.py: A Python library containing all from-scratch implementations of algorithms used (PCA, KNN, Softmax, RandomForest, KMeans, etc.).

* End-Semester Project Report_MNIST Digit Classifier.pdf: The full PDF report detailing our system architecture, hyper-parameter tuning, optimization steps, and final observations.

* MNIST_train.csv: Training data (required by main.py).

* MNIST_validation.csv: Validation data (required by main.py).

* README.md: This file, explaining the project.

---

## 📦 Installation & Dependencies

* Only numpy and scipy are required to run this project.
``` pip install numpy scipy ```
---

## ▶️ Running the Code

All required data files are included in this repository. All experiments should be runnable from the command line.

### A. Command-line (recommended for grading)

* To run my final model, reproduce our results, and get the final validation score, simply navigate to the repository's root folder and run:
  ``` python main.py ```
* The script will load the data, apply PCA, train the KNN model, and print the final Macro F1 Score and total runtime, which was 5.95 seconds on the test machine.
---

## 🧾 Authors

**<M Shivaarchitha Vudutha, DA24B052>**, IIT Madras (2024–28)
