import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import datetime
from datetime import timedelta
from tqdm import tqdm
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import os


from ecg_utils import prepare_mimicecg
from timeseries_utils import reformat_as_memmap

import argparse
from pathlib import Path






# Convert signals into numpy 
zip_file_path = Path('mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip') # path to mimic-ecg zip
target_path = Path('data/memmap/') # desired output path
df,_,_,_=prepare_mimicecg(zip_file_path, target_folder=target_path)


# Reformat as memmap for fast access
reformat_as_memmap(df, 
                   target_path/"memmap.npy", 
                   data_folder=target_path, 
                   annotation=False, 
                   max_len=0, 
                   delete_npys=True, 
                   col_data="data", 
                   col_lbl=None, 
                   batch_length=0, 
                   skip_export_signals=False)









# load and basic cleaning
record_list = pd.read_csv('record_list.csv') 
record_list['data'] = record_list.index
patients = pd.read_csv('patients.csv.gz', compression='gzip') 
record_list = record_list.merge(patients[['subject_id','gender','anchor_age']], on='subject_id')
record_list['gender'] = record_list['gender'].apply(lambda x: 1 if x=='M' else 0)
df_labtitles = pd.read_csv('d_labitems.csv.gz', compression='gzip') 
df_labevents = pd.read_csv('labevents.csv.gz', compression='gzip', low_memory=False)
df_labs = pd.merge(df_labevents, df_labtitles, on='itemid')
df_labevents = None
df_labs = df_labs[df_labs['subject_id'].isin(record_list['subject_id'].unique())]
df_labs.dropna(subset='valuenum', inplace=True)
df_labs.dropna(subset='charttime', inplace=True)
df_labs.dropna(subset='label', inplace=True)
df_labs = df_labs[~df_labs['label'].isin(['other'])]
record_list['ecg_time'] = pd.to_datetime(record_list['ecg_time'])
df_labs['charttime'] = pd.to_datetime(df_labs['charttime'])

# add ecg features
machine_measurements_dic = pd.read_csv('machine_measurements_data_dictionary.csv') 
machine_measurements = pd.read_csv('machine_measurements.csv') 
machine_measurements = machine_measurements[['study_id','rr_interval','p_onset','p_end','qrs_onset','qrs_end','t_end','p_axis','qrs_axis','t_axis']]
record_list = record_list.merge(machine_measurements, on='study_id')

# where we can't make labels
mask_zero_low_high = (df_labs['ref_range_lower'] == 0) & (df_labs['ref_range_upper'] == 0)
df_labs = df_labs[~mask_zero_low_high]

# not documented anywhere
df_labs = df_labs[~df_labs['label'].isin(['H', 'I', 'L'])] 

# add race
def map_race(row):
    if 'WHITE' in row:
        return 'white'
    elif 'BLACK' in row:
        return 'black'
    elif 'HISPANIC' in row:
        return 'hispanic'
    elif 'ASIAN' in row:
        return 'asian'
    else:
        return 'other'
    
df_hosp_admission = pd.read_csv('/user/leal6863/Groningen/mimic_preprocess/admissions.csv.gz')
df_hosp_admission['race_mapped'] = df_hosp_admission['race'].apply(map_race)
df_ed_admissions = pd.read_csv('edstays.csv.gz') 
df_ed_admissions['race_mapped'] = df_ed_admissions['race'].apply(map_race)
df_hosp_admission_filtered = df_hosp_admission[['subject_id', 'race_mapped']]
df_ed_admissions_filtered = df_ed_admissions[['subject_id', 'race_mapped']]
df_combined = pd.concat([df_hosp_admission_filtered, df_ed_admissions_filtered])
df_unique_subjects = df_combined.drop_duplicates(subset='subject_id').reset_index(drop=True)
record_list = record_list.merge(df_unique_subjects, on='subject_id', how='left')
record_list.rename(columns={'race_mapped': 'race'}, inplace=True)

# stratification based on diagnoses, gender, age
df_diags = pd.read_csv('records_w_diag_icd10.csv')
record_list = record_list.merge(df_diags[['study_id','strat_fold']], on='study_id', how='left')

# adjust these units
df_labs.loc[df_labs['label'] == 'pH', 'valueuom'] = 'units' # same consistency
df_labs.loc[df_labs['label'] == 'Specific Gravity', 'valueuom'] = ' ' # same consistency
df_labs.loc[df_labs['label'] == 'MCHC', 'valueuom'] = 'g/dL' # same consistency
df_labs = df_labs[~((df_labs['label'] == 'Absolute Lymphocyte Count') & (df_labs['valueuom'] == '#/uL'))] # different and minority
df_labs.loc[df_labs['label'] == 'Protein/Creatinine Ratio', 'valueuom'] = 'mg/mg' # same consistency
df_labs = df_labs[~((df_labs['label'] == 'D-Dimer') & (df_labs['valueuom'] == 'ng/mL DDU'))] # different and minority
df_labs.loc[df_labs['label'] == 'D-Dimer', 'valueuom'] = 'ng/mL' # same for consistency
df_labs.loc[df_labs['label'] == 'Thyroid Stimulating Hormone', 'valueuom'] = 'uIU/mL' # same for consistency
df_labs.loc[df_labs['label'] == 'Cortisol', 'valueuom'] = 'ug/dL' # same for consistency
df_labs.loc[df_labs['label'] == 'RDW-SD', 'valueuom'] = 'fL' # same for consistency
df_labs.loc[df_labs['label'] == 'Albumin/Creatinine, Urine', 'valueuom'] = 'mg/g' # same for consistency
df_labs.loc[df_labs['label'] == 'C-Reactive Protein', 'valueuom'] = 'mg/L' # same for consistency
df_labs.loc[df_labs['label'] == 'eAG', 'valueuom'] = 'mg/dL' # same for consistency
df_labs.loc[df_labs['label'] == 'Absolute CD3 Count', 'valueuom'] = '#/uL' # same for consistency
df_labs.loc[df_labs['label'] == 'Absolute CD4 Count', 'valueuom'] = '#/uL' # same for consistency
df_labs.loc[df_labs['label'] == 'Absolute CD8 Count', 'valueuom'] = '#/uL' # same for consistency
df_labs.loc[df_labs['label'] == 'CD4/CD8 Ratio', 'valueuom'] = 'ratio' # same for consistency
df_labs.loc[df_labs['label'] == 'WBC Count', 'valueuom'] = 'K/uL' # same for consistency
df_labs.loc[df_labs['label'] == 'Prostate Specific Antigen', 'valueuom'] = 'mg/dL' # same for consistency

# keep fluid consistency
fluid_counts = df_labs.groupby(['label', 'fluid']).size().reset_index(name='count')
idx = fluid_counts.groupby('label')['count'].idxmax()
top_fluid_counts = fluid_counts.loc[idx]
df_labs = df_labs.merge(top_fluid_counts[['label', 'fluid']], on=['label', 'fluid'])

# these values won't make the cut 
label_counts = df_labs['label'].value_counts()
labels_to_keep = label_counts[label_counts >= 200].index
df_labs = df_labs[df_labs['label'].isin(labels_to_keep)]

# get classes (low,normal,high), drop lab values that have only 1.
median_values = df_labs.groupby('label')[['ref_range_lower', 'ref_range_upper']].median().reset_index()
df_labs = df_labs.merge(median_values, on='label', suffixes=('', '_median'))

def classify_value(row):
    if row['valuenum'] < row['ref_range_lower_median']:
        return 'low'
    elif row['valuenum'] > row['ref_range_upper_median']:
        return 'high'
    else:
        return 'normal'
    
new_rows = []

for lab in tqdm(df_labs['label'].unique()):
    df_lab = df_labs[df_labs['label'] == lab].copy()
    df_lab['class'] = df_lab.apply(classify_value, axis=1)
    
    if len(df_lab['class'].unique())>1: # more than 1 class, can compute 5% yet 
        new_rows.append(df_lab)

df_labs = pd.concat(new_rows, ignore_index=True)



label_classes = {}
for label in tqdm(sorted(df_labs['label'].unique())):
    unique_classes = df_labs[df_labs['label'] == label]['class'].unique()
    label_classes[label] = [cls for cls in unique_classes if cls != 'normal']

all_final_labels = [f"{label}_{cls}" for label, classes in label_classes.items() for cls in classes]


lbl_itos_estimation = all_final_labels
lbl_itos_monitoring = all_final_labels 


# ECG feaures cleaning

# degrees
record_list.loc[(record_list['qrs_axis'] < -360) | (record_list['qrs_axis'] > 360), 'qrs_axis'] = np.nan
record_list.loc[(record_list['t_axis'] < -360) | (record_list['t_axis'] > 360), 't_axis'] = np.nan
record_list.loc[(record_list['p_axis'] < -360) | (record_list['p_axis'] > 360), 'p_axis'] = np.nan

# msec
record_list.loc[(record_list['p_onset'] < 0) | (record_list['p_onset'] > 5000), 'p_onset'] = np.nan
record_list.loc[(record_list['p_end'] < 0) | (record_list['p_end'] > 5000), 'p_end'] = np.nan
record_list.loc[(record_list['qrs_onset'] < 0) | (record_list['qrs_onset'] > 5000), 'qrs_onset'] = np.nan
record_list.loc[(record_list['qrs_end'] < 0) | (record_list['qrs_end'] > 5000), 'qrs_end'] = np.nan
record_list.loc[(record_list['t_end'] < 0) | (record_list['t_end'] > 5000), 't_end'] = np.nan
record_list.loc[(record_list['rr_interval'] < 0) | (record_list['rr_interval'] > 5000), 'rr_interval'] = np.nan

save_path = 'saved_data/'
os.makedirs(save_path, exist_ok=True)

np.save('saved_data/lbl_itos_estimation.npy', lbl_itos_estimation)
np.save('saved_data/lbl_itos_monitoring.npy', lbl_itos_monitoring)

df_labs.to_pickle('saved_data/df_labs.pkl')
record_list.to_pickle('saved_data/record_list.pkl')



figures_dir = "saved_labs_patients"
os.makedirs(figures_dir, exist_ok=True)


# saves computations
for patient in tqdm(df_labs['subject_id'].unique()):
    df_lab_patient = df_labs[df_labs['subject_id']==patient]
    df_lab_patient.to_pickle(f'saved_labs_patients/{patient}.pkl')
    
    
    
df_labs = pd.read_pickle('saved_data/df_labs.pkl')
record_list = pd.read_pickle('saved_data/record_list.pkl')
lbl_itos_estimation = np.load('saved_data/lbl_itos_estimation.npy')
lbl_itos_monitoring = np.load('saved_data/lbl_itos_monitoring.npy')




dummies = pd.get_dummies(record_list['race'].astype(str), prefix='race', dummy_na=False)
record_list = pd.concat([record_list, dummies], axis=1)
record_list.loc[record_list['race'].isna(), dummies.columns] = np.nan
record_list.drop(['race','race_nan'], axis=1, inplace=True)



df_vital = pd.read_csv('vitalsign.csv.gz') 
df_vital['charttime'] = pd.to_datetime(df_vital['charttime'])
df_vital = df_vital.iloc[:,:-2]
df_vital = df_vital[df_vital['subject_id'].isin(record_list['subject_id'].unique())]

df_vital.loc[df_vital['temperature'] < 50, 'temperature'] = np.nan # 53 minimum recorded
df_vital.loc[df_vital['temperature'] > 150, 'temperature'] = np.nan # 115.7 maximum recorded
df_vital.loc[df_vital['heartrate'] > 700, 'heartrate'] = np.nan # 600 maximum recorded
df_vital.loc[df_vital['resprate'] > 300, 'resprate'] = np.nan # normal 20, athletes 50
df_vital.loc[df_vital['o2sat'] > 100, 'o2sat'] = np.nan # can't be negative nor more than 100
df_vital.loc[df_vital['o2sat'] < 0, 'o2sat'] = np.nan # can't be negative nor more than 100
df_vital.loc[df_vital['dbp'] > 500, 'dbp'] = np.nan # max recorded 370
df_vital.loc[df_vital['sbp'] > 500, 'sbp'] = np.nan # max recorded 360



def closest_non_nan(series, times, ecg_time):
    """ Return the closest non-NaN value in the series to ecg_time, or NaN if all values are NaN """
    non_nan_series = series.dropna()
    if non_nan_series.empty:
        return np.nan
    time_diffs = abs(times[non_nan_series.index] - ecg_time)
    closest_index = time_diffs.idxmin()
    return non_nan_series.loc[closest_index]

    out_temperature = []
    out_heartrate = []
    out_resprate = []
    out_o2sat = []
    out_dbp = []
    out_sbp = []

    for _, row in tqdm(record_list.iterrows(), total=len(record_list)):
        patient = row['subject_id']
        ecg_time = row['ecg_time']
        df_patient = df_vital[df_vital['subject_id'] == patient]

        df_patient_within = df_patient.loc[(df_patient['charttime'] >= (ecg_time - pd.Timedelta(minutes=30))) & 
                                           (df_patient['charttime'] <= (ecg_time + pd.Timedelta(minutes=30)))]

        if df_patient_within.empty:
            out_temperature.append(np.nan)
            out_heartrate.append(np.nan)
            out_resprate.append(np.nan)
            out_o2sat.append(np.nan)
            out_dbp.append(np.nan)
            out_sbp.append(np.nan)
        else:
            charttimes = df_patient_within['charttime']
            temperature = closest_non_nan(df_patient_within['temperature'], charttimes, ecg_time)
            heartrate = closest_non_nan(df_patient_within['heartrate'], charttimes, ecg_time)
            resprate = closest_non_nan(df_patient_within['resprate'], charttimes, ecg_time)
            o2sat = closest_non_nan(df_patient_within['o2sat'], charttimes, ecg_time)
            sbp = closest_non_nan(df_patient_within['sbp'], charttimes, ecg_time)
            dbp = closest_non_nan(df_patient_within['dbp'], charttimes, ecg_time)

            out_temperature.append(temperature)
            out_heartrate.append(heartrate)
            out_resprate.append(resprate)
            out_o2sat.append(o2sat)
            out_dbp.append(dbp)
            out_sbp.append(sbp)


            
            
record_list['temperature'] = out_temperature
record_list['heartrate'] = out_heartrate
record_list['resprate'] = out_resprate
record_list['o2sat'] = out_o2sat
record_list['dbp'] = out_dbp
record_list['sbp'] = out_sbp








omr = pd.read_csv('omr.csv.gz') 
omr['result_value'] = pd.to_numeric(omr['result_value'], errors='coerce')
omr['chartdate'] = pd.to_datetime(omr['chartdate'])
omr = omr[omr['result_name'].isin(['BMI (kg/m2)','Height (Inches)','Weight (Lbs)'])]


omr = omr[omr['subject_id'].isin(record_list['subject_id'].unique())]
omr.dropna(subset=['result_value'], inplace=True)
omr.loc[omr['result_name'] == 'Height (Inches)', 'result_value'] *= 2.54
omr.loc[omr['result_name'] == 'Height (Inches)', 'result_name'] = 'Height (cm)'
omr.loc[omr['result_name'] == 'Weight (Lbs)', 'result_value'] *= 0.453592
omr.loc[omr['result_name'] == 'Weight (Lbs)', 'result_name'] = 'Weight (kg)'


conditions = [
    (omr['result_name'] == 'BMI (kg/m2)') & (omr['result_value'] > 100),
    (omr['result_name'] == 'Weight (kg)') & (omr['result_value'] > 400),
    (omr['result_name'] == 'Height (cm)') & (omr['result_value'] > 400),
    (omr['result_name'] == 'Weight (kg)') & (omr['result_value'] < 20),
    (omr['result_name'] == 'Height (cm)') & (omr['result_value'] < 60)
]

for condition in conditions:
    omr.loc[condition, 'result_value'] = np.nan
    
omr.dropna(subset=['result_value'], inplace=True)


out_bmi = []
out_weight = []
out_height = []

for _, row in tqdm(record_list.iterrows(), total=len(record_list)):
    patient = row['subject_id']
    intime = row['ecg_time']
    
    df_patient = omr[omr['subject_id'] == patient]
    df_patient_within = df_patient.loc[(df_patient['chartdate'] >= (intime - pd.Timedelta(days=30))) & 
                                       (df_patient['chartdate'] <= (intime + pd.Timedelta(days=30)))]
    
    if df_patient_within.empty:
        out_bmi.append(np.nan)
        out_weight.append(np.nan)
        out_height.append(np.nan)
    else:
        # Find the closest BMI to ecg_time
        bmi_rows = df_patient_within[df_patient_within['result_name'] == 'BMI (kg/m2)']
        if not bmi_rows.empty:
            closest_bmi = bmi_rows.iloc[(bmi_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            out_bmi.append(closest_bmi)
        else:
            out_bmi.append(np.nan)
        
        # Find the closest Weight to ecg_time
        weight_rows = df_patient_within[df_patient_within['result_name'] == 'Weight (kg)']
        if not weight_rows.empty:
            closest_weight = weight_rows.iloc[(weight_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            out_weight.append(closest_weight)
        else:
            out_weight.append(np.nan)
        
        # Find the closest Height to ecg_time
        height_rows = df_patient_within[df_patient_within['result_name'] == 'Height (cm)']
        if not height_rows.empty:
            closest_height = height_rows.iloc[(height_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            out_height.append(closest_height)
        else:
            out_height.append(np.nan)
            
record_list['bmi'] = out_bmi
record_list['weight'] = out_weight
record_list['height'] = out_height



record_list.to_pickle('saved_data/record_list_updated.pkl')
record_list = pd.read_pickle('saved_data/record_list_updated.pkl')

def save_results(index, out_estimation, out_monitoring30, out_monitoring60, out_monitoring120):
    """ Save the results to pickle files """
    with open(f'saved_data/out_estimation_{index}.pkl', 'wb') as f:
        pkl.dump(out_estimation, f)
    with open(f'saved_data/out_monitoring30_{index}.pkl', 'wb') as f:
        pkl.dump(out_monitoring30, f)
    with open(f'saved_data/out_monitoring60_{index}.pkl', 'wb') as f:
        pkl.dump(out_monitoring60, f)
    with open(f'saved_data/out_monitoring120_{index}.pkl', 'wb') as f:
        pkl.dump(out_monitoring120, f)
        
        
        
        
out_estimation = []
out_monitoring30 = []
out_monitoring60 = []
out_monitoring120 = []

for index, record_row in tqdm(enumerate(record_list.itertuples()), total=len(record_list)):
    
    subject_id = record_row.subject_id
    ecg_time = record_row.ecg_time
    
    try:
        
        df_lab_p = pd.read_pickle(f'saved_labs_patients/{subject_id}.pkl')
        
        relevant_lab_values_est = df_lab_p[(df_lab_p['charttime'] >= ecg_time - timedelta(minutes=60)) & (df_lab_p['charttime'] <= ecg_time + timedelta(minutes=60))].copy()
        relevant_lab_values_mon30 = df_lab_p[(df_lab_p['charttime'] >= ecg_time) & (df_lab_p['charttime'] <= ecg_time + timedelta(minutes=30))].copy()
        relevant_lab_values_mon60 = df_lab_p[(df_lab_p['charttime'] >= ecg_time) & (df_lab_p['charttime'] <= ecg_time + timedelta(minutes=60))].copy()
        relevant_lab_values_mon120 = df_lab_p[(df_lab_p['charttime'] >= ecg_time) & (df_lab_p['charttime'] <= ecg_time + timedelta(minutes=120))].copy()
        
        
    except FileNotFoundError:
        relevant_lab_values_est = pd.DataFrame()
        relevant_lab_values_mon30 = pd.DataFrame()
        relevant_lab_values_mon60 = pd.DataFrame()
        relevant_lab_values_mon120 = pd.DataFrame()
        
        
    except Exception as e:
        logging.error(f'An error occurred for subject_id {subject_id}: {e}')
        continue
                

    # closest to ecg, no more than 60 min from lab value
    final_label_class_est = [-999] * len(lbl_itos_estimation)
    
    if not relevant_lab_values_est.empty:
        relevant_first = relevant_lab_values_est.assign(time_diff=abs(relevant_lab_values_est['charttime'] - ecg_time))
        relevant_first = relevant_first.sort_values(by=['label', 'time_diff']).drop_duplicates(subset='label', keep='first')
        
        for i, final_label_name in enumerate(lbl_itos_estimation):  
            label,cls = final_label_name.rsplit('_',1)
            if label in relevant_first['label'].values: 
                relevant_first_label = relevant_first[relevant_first['label']==label]
                if cls in relevant_first_label['class'].values: 
                    final_label_class_est[i] = 1
                else:
                    final_label_class_est[i] = 0
                    
                    
    # create monitoring final label 30
    final_label_class_mon30 = [-999] * len(lbl_itos_monitoring)
    
    if not relevant_lab_values_mon30.empty:
        for i, final_label_name in enumerate(lbl_itos_monitoring):
            label,cls = final_label_name.rsplit('_',1)
            if label in relevant_lab_values_mon30['label'].values:
                relevant_label = relevant_lab_values_mon30[relevant_lab_values_mon30['label']==label]
                if 'low' in relevant_label['class'].values and 'high' in relevant_label['class'].values:
                    final_label_class_mon30[i] = -999
                else:
                    if cls in relevant_label['class'].values:
                        final_label_class_mon30[i] = 1
                    else:
                        final_label_class_mon30[i] = 0

                        
                        
    # create monitoring final label 60
    final_label_class_mon60 = [-999] * len(lbl_itos_monitoring)
    
    if not relevant_lab_values_mon60.empty:
        for i, final_label_name in enumerate(lbl_itos_monitoring):
            label,cls = final_label_name.rsplit('_',1)
            if label in relevant_lab_values_mon60['label'].values:
                relevant_label = relevant_lab_values_mon60[relevant_lab_values_mon60['label']==label]
                if 'low' in relevant_label['class'].values and 'high' in relevant_label['class'].values:
                    final_label_class_mon60[i] = -999
                else:
                    if cls in relevant_label['class'].values:
                        final_label_class_mon60[i] = 1
                    else:
                        final_label_class_mon60[i] = 0
                        
                
    # create monitoring final label 120
    final_label_class_mon120 = [-999] * len(lbl_itos_monitoring)
    
    if not relevant_lab_values_mon120.empty:
        for i, final_label_name in enumerate(lbl_itos_monitoring):
            label,cls = final_label_name.rsplit('_',1)
            if label in relevant_lab_values_mon120['label'].values:
                relevant_label = relevant_lab_values_mon120[relevant_lab_values_mon120['label']==label]
                if 'low' in relevant_label['class'].values and 'high' in relevant_label['class'].values:
                    final_label_class_mon120[i] = -999
                else:
                    if cls in relevant_label['class'].values:
                        final_label_class_mon120[i] = 1
                    else:
                        final_label_class_mon120[i] = 0


    out_estimation.append(final_label_class_est)
    out_monitoring30.append(final_label_class_mon30)
    out_monitoring60.append(final_label_class_mon60)
    out_monitoring120.append(final_label_class_mon120)
    
    if (index + 1) % 50000 == 0:
        save_results(index + 1, out_estimation, out_monitoring30, out_monitoring60, out_monitoring120)
        
        # empty the lists
        out_estimation = []
        out_monitoring30 = []
        out_monitoring60 = []
        out_monitoring120 = []

# Save remaining data if any
if out_estimation:
    save_results('final', out_estimation, out_monitoring30, out_monitoring60, out_monitoring120)
    
    
    
suffixes = [i+'.pkl' for i in [str(i) for i in np.arange(50000,800000,50000)] + ['final']]

files_estimation = ['saved_data/out_estimation_'] * 16
files_monitoring30 = ['saved_data/out_monitoring30_'] * 16
files_monitoring60 = ['saved_data/out_monitoring60_'] * 16
files_monitoring120 = ['saved_data/out_monitoring120_'] * 16


files_estimation = [f+s for f,s in zip(files_estimation, suffixes)]
files_monitoring30 = [f+s for f,s in zip(files_monitoring30, suffixes)]
files_monitoring60 = [f+s for f,s in zip(files_monitoring60, suffixes)]
files_monitoring120 = [f+s for f,s in zip(files_monitoring120, suffixes)]

def load(files):
    concatenated_list = []
    for file in files:
        with open(file, 'rb') as f:
            concatenated_list.extend(pkl.load(f))
    return concatenated_list

out_estimation = load(files_estimation)
out_monitoring30 = load(files_monitoring30)
out_monitoring60 = load(files_monitoring60)
out_monitoring120 = load(files_monitoring120)

record_list['label_estimation'] = out_estimation
record_list['label_monitoring30'] = out_monitoring30
record_list['label_monitoring60'] = out_monitoring60
record_list['label_monitoring120'] = out_monitoring120

lbl_itos = np.load('saved_data/lbl_itos_monitoring.npy')


def get_to_drop(column_name, lbl_itos):

    column_name = column_name
    lbls_to_drop = []
    index_to_drop = []

    # Iterate through each label
    for i, lbl in tqdm(enumerate(lbl_itos)):
        should_drop = False

        # Iterate through each fold to check the label distribution
        for fold in [18,19]: # val and test 10 minimum selection criteria.
            fold_data = record_list[record_list['strat_fold'] == fold]
            label_data = np.stack(fold_data[column_name].values)[:, i]
            uniques, counts = np.unique(label_data, return_counts=True)

            try:
                where0 = np.where(uniques == 0)[0][0]
                where1 = np.where(uniques == 1)[0][0]
                counts0 = counts[where0]
                counts1 = counts[where1]

                # Check if either count is less than 10
                if counts0 < 10 or counts1 < 10:
                    should_drop = True
                    break  # No need to check other folds for this label, move to the next label

            except IndexError:  # Handle the case where 0 or 1 is not present
                should_drop = True
                break

        # If the label should be dropped, add to lists
        if should_drop:
            lbls_to_drop.append(lbl)
            index_to_drop.append(i)
            
    
    return lbls_to_drop, index_to_drop


lbls_to_drop_estimation, index_to_drop_estimation = get_to_drop('label_estimation', lbl_itos)
lbls_to_drop_monitoring30, index_to_drop_monitoring30 = get_to_drop('label_monitoring30', lbl_itos)
lbls_to_drop_monitoring60, index_to_drop_monitoring60 = get_to_drop('label_monitoring60', lbl_itos)
lbls_to_drop_monitoring120, index_to_drop_monitoring120 = get_to_drop('label_monitoring120', lbl_itos)


lbls_to_keep_estimation = [i for i in lbl_itos if i not in lbls_to_drop_estimation]
lbls_to_keep_monitoring30 = [i for i in lbl_itos if i not in lbls_to_drop_monitoring30]
lbls_to_keep_monitoring60 = [i for i in lbl_itos if i not in lbls_to_drop_monitoring60]
lbls_to_keep_monitoring120 = [i for i in lbl_itos if i not in lbls_to_drop_monitoring120]

column_data_estimation = np.stack(record_list['label_estimation'].values)
column_data_monitoring30 = np.stack(record_list['label_monitoring30'].values)
column_data_monitoring60 = np.stack(record_list['label_monitoring60'].values)
column_data_monitoring120 = np.stack(record_list['label_monitoring120'].values)

new_column_data_estimation = column_data_estimation[:,[i for i in range(len(lbl_itos)) if i not in index_to_drop_estimation]]
new_column_data_monitoring30 = column_data_monitoring30[:,[i for i in range(len(lbl_itos)) if i not in index_to_drop_monitoring30]]
new_column_data_monitoring60 = column_data_monitoring60[:,[i for i in range(len(lbl_itos)) if i not in index_to_drop_monitoring60]]
new_column_data_monitoring120 = column_data_monitoring120[:,[i for i in range(len(lbl_itos)) if i not in index_to_drop_monitoring120]]

record_list['final_label_estimation'] = [list(row) for row in new_column_data_estimation]
record_list['final_label_monitoring30'] = [list(row) for row in new_column_data_monitoring30]
record_list['final_label_monitoring60'] = [list(row) for row in new_column_data_monitoring60]
record_list['final_label_monitoring120'] = [list(row) for row in new_column_data_monitoring120]

record_list.drop(['label_estimation',
                  'label_monitoring30',
                  'label_monitoring60',
                  'label_monitoring120'], axis=1, inplace=True)

np.save('saved_data/lbl_itos_estimation.npy', lbls_to_keep_estimation)
np.save('saved_data/lbl_itos_monitoring30.npy', lbls_to_keep_monitoring30)
np.save('saved_data/lbl_itos_monitoring60.npy', lbls_to_keep_monitoring60)
np.save('saved_data/lbl_itos_monitoring120.npy', lbls_to_keep_monitoring120)


def has_valid_element(lst):
    return any(x != -999 for x in lst)

filtered_record_list = record_list[
    record_list['final_label_estimation'].apply(has_valid_element) |
    record_list['final_label_monitoring30'].apply(has_valid_element) |
    record_list['final_label_monitoring60'].apply(has_valid_element) |
    record_list['final_label_monitoring120'].apply(has_valid_element)
]

record_list = None
def aggregate_lists(row):
    return row['final_label_estimation'] + row['final_label_monitoring30'] + row['final_label_monitoring60'] + row['final_label_monitoring120']

filtered_record_list['aggregated_label'] = filtered_record_list.apply(aggregate_lists, axis=1)

filtered_record_list.drop(['final_label_estimation',
                           'final_label_monitoring30', 
                           'final_label_monitoring60',
                           'final_label_monitoring120'], axis=1, inplace=True)




np.save('saved_data/all_features.npy', all_features)
filtered_record_list.to_pickle('saved_data/df_memmap.pkl')

