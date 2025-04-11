## This is the official repository for <ins>CardioLab</ins>. A machine and deep learning framework for the estimation and monitoring of laboratory values throught ECG data.

CardioLab have been proposed in a main manuscript:
   
**CardioLab: Laboratory Values Estimation and Monitoring from Electrocardiogram Signals - A Deep-Multimodal Approach** 


In terms of ECG data and clinical settings, our CinC manuscript investigate only abnormalities estimation (current) task with ECG tabular features, whereas our second manuscript investigate abnormalities estimation (current) as well as abnormalities monitoring (future) using ECG raw waveforms instead.


## Clinical Setting

![alt text](https://github.com/AI4HealthUOL/KardioLab/blob/main/reports/abstract.png?style=centerme)

 
- A) Demonstrates the overall predictive workflow used in the study, where for model inputs we use ECG waveforms, demographics, biometrics, and vital signs, in a binary classification setting to predict abnormal laboratory values.

- B) Demonstrates the estimation task, where for feature space we sample the closest vital signs within 30 minutes of the ECG record, and the target is the closest laboratory value within 60 minutes.

- C) Demonstrates the monitoring task, where the feature space also includes the closest vital signs within 30 minutes of the ECG record, and the target is the presence of any abnormal laboratory value within a defined future time horizon, for which we investigated 30, 60, and 120 minutes.






