## This is the official repository for <ins>CardioLab</ins>. A machine and deep learning framework for the estimation and monitoring of laboratory values throught ECG data.

CardioLab have been proposed in two main manuscrips:

1. **CardioLab: Laboratory Values Estimation from Electrocardiogram Features - An Exploratory Study.** <ins>Accepted by the international conference of computing in cardiology (CinC) 2024.</ins> [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2407.18629)
   
2. **CardioLab: Laboratory Values Estimation and Monitoring from Electrocardiogram Signals - A Deep-Multimodal Approach** [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2411.14886)


In terms of ECG data and clinical settings, our CinC manuscript investigate only abnormalities estimation (current) task with ECG tabular features, whereas our second manuscript investigate abnormalities estimation (current) as well as abnormalities monitoring (future) using ECG raw waveforms instead.


## Clinical Setting

![alt text](https://github.com/AI4HealthUOL/KardioLab/blob/main/reports/abstract.png?style=centerme)

 
- A) Demonstrates the overall predictive workflow used in the study, where for model inputs we use ECG waveforms, demographics, biometrics, and vital signs, in a binary classification setting to predict abnormal laboratory values.

- B) Demonstrates the estimation task, where for feature space we sample the closest vital signs within 30 minutes of the ECG record, and the target is the closest laboratory value within 60 minutes.

- C) Demonstrates the monitoring task, where the feature space also includes the closest vital signs within 30 minutes of the ECG record, and the target is the presence of any abnormal laboratory value within a defined future time horizon, for which we investigated 30, 60, and 120 minutes.



## References

```bibtex
@misc{alcaraz2024cardiolablaboratoryvaluesestimation,
      title={CardioLab: Laboratory Values Estimation from Electrocardiogram Features -- An Exploratory Study}, 
      author={Juan Miguel Lopez Alcaraz and Nils Strodthoff},
      year={2024},
      eprint={2407.18629},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2407.18629}, 
}
```

```bibtex
@misc{alcaraz2024cardiolablaboratoryvaluesestimation,
      title={CardioLab: Laboratory Values Estimation and Monitoring from Electrocardiogram Signals -- A Multimodal Deep Learning Approach}, 
      author={Juan Miguel Lopez Alcaraz and Nils Strodthoff},
      year={2024},
      eprint={2411.14886},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2411.14886}, 
}
```



