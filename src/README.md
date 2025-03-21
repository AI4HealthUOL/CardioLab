# Replicating the Experiments

This repository provides scripts to preprocess ECG data, train models, and analyze results. Follow the steps below to replicate the experiments.

## 1. Download Required Datasets
Before running the scripts, download the following files and place them in the root directory of this repository:

- **records_w_diag_icd10.csv** from [PhysioNet](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/)
- **machine_measurements.csv** from [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **ECG_ViEW_II_for_CVS.zip** (do not unzip) from [ECG ViEW II](http://ecgview.org/)

## 2. Install Dependencies
Ensure you have Python installed (recommended version: 3.8+). Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 3. Run Preprocessing
First, preprocess the data by running:

```bash
python preprocessing.py
```

## 4. Train the Model
After preprocessing is complete, train the model by running:

```bash
python main.py
```

## 5. Output and Results
The results, including model performance metrics and analyses, will be saved automatically in the `here/` directory.

If you encounter any issues, verify the dataset paths and ensure dependencies are correctly installed.

---

For further inquiries, please open an issue.
