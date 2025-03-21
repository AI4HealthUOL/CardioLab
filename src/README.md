# Replicating the Experiments

This repository provides scripts to preprocess ECG data, train models, and analyze results. Follow the steps below to replicate the experiments.

## 1. Download Required Datasets
Before running the scripts, download the following files and place them in the root directory of this repository:



- **patients.csv.gz** from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
- **d_labitems.csv.gz** from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
- **labevents.csv.gz** from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
- **omr.csv.gz** from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)

- **edstays.csv.gz** from [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/)
- **vitalsign.csv.gz** from [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/)

- **records_w_diag_icd10.csv** from [MIMIC-IV-ECG-ICD](https://www.physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/)

- **record_list.csv** from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **machine_measurements.csv** from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **machine_measurements_data_dictionary.csv** from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip** from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
  

## 2. Run Preprocessing
First, preprocess the data by running:

```bash
python preprocessing.py
```

## 3. Train the Model
After preprocessing is complete, train the model by running:

```bash
python main_all.py --config config/config_supervised_multimodal_labvalues_s4.yaml
```

## 4. Output and Results
The results, including model performance metrics and analyses, will be saved automatically in the `here/` directory.

If you encounter any issues, verify the dataset paths and ensure dependencies are correctly installed.

---

For further inquiries, please open an issue.
