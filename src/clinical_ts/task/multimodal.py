from ..template_model import SSLModel
from ..template_modules import TaskConfig
from ..data.time_series_dataset_utils import load_dataset
from ..data.time_series_dataset_transforms import Transform
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings

#disable performance warning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class MultimodalModel(SSLModel):
    '''class for multimodal tasks'''
   
    def preprocess_dataset(self,dataset_kwargs):
        df_mapped, lbl_itos, mean, std = load_dataset(Path(dataset_kwargs.path))

        if(self.hparams.loss.loss_type=="supervised" and dataset_kwargs.name.startswith("mimic_labvalues")):
            
            #reformat race
            lbl_itos_ethnicity=['race_asian', 'race_black', 'race_hispanic', 'race_other', 'race_white']

            df_mapped["race"]=df_mapped.apply(lambda row: np.where([row[c] for c in lbl_itos_ethnicity])[0],axis=1)
            df_mapped["race"]=df_mapped["race"].apply(lambda x:x[0] if len(x)==1 else -1)
            df_mapped["race_nan"]=(df_mapped["race"]==-1)
            if(self.hparams.task.impute_nans):
                df_mapped["race"] = df_mapped["race"].replace(-1, df_mapped[(df_mapped["race"] != -1)&(df_mapped.strat_fold<18)]["race"].median())# median imput missing race
            df_mapped.drop(lbl_itos_ethnicity,axis=1,inplace=True)

            #prepare cat and cont features
            cat_features = ['gender','race']
            cat_features_m =['race_nan', 'temperature_nan', 'o2sat_nan', 'weight_nan', 'bmi_nan', 'sbp_nan', 'dbp_nan', 'resprate_nan', 'heartrate_nan', 'height_nan']

            cont_features = ['resprate', 'o2sat', 'anchor_age', 'dbp', 'heartrate', 'sbp', 'temperature',  'height', 'bmi', 'weight']#'rr_interval', 'qrs_onset', 'qrs_axis', 'p_axis', 't_axis', 'p_end', 'p_onset', 'qrs_end', 't_end'

            if(self.hparams.task.impute_nans):
                input_cols = cat_features + cont_features
                #grab training set medians and identify columns with nans
                df_train= df_mapped[df_mapped.strat_fold<18]
                train_medians= df_train[input_cols].median().to_dict()
                train_nans = [l for l,c in df_train[input_cols].isna().sum().to_dict().items() if c>0]

                #impute nans through medians introduce additional column _nan (if desired)
                for c in train_nans:
                    if(self.hparams.task.introduce_nan_columns):
                        df_mapped[c+"_nan"]=0
                        df_mapped.loc[df_mapped[c].isna(),c+"_nan"]=1
                    df_mapped.loc[df_mapped[c].isna(),c]=train_medians[c]

                #defragment
                df_mapped = df_mapped.copy()

            if(self.hparams.task.introduce_nan_columns):
                cat_features += cat_features_m
            
            cat_features_dim = [len(df_mapped[c].unique()) for c in cat_features]
            
            df_mapped["cat_features"]=df_mapped[cat_features].values.tolist()
            df_mapped.drop(cat_features,axis=1,inplace=True)
            df_mapped["cont_features"]=df_mapped[cont_features].values.tolist()
            df_mapped.drop(cont_features,axis=1,inplace=True)
    
            def replace_nan(arr):
                return np.where(np.array(arr)==-999.0, np.nan, np.array(arr))
            df_mapped["label"]=df_mapped["aggregated_label"].apply(lambda arr:replace_nan(arr))
            df_mapped.drop("aggregated_label",axis=1,inplace=True)

            return df_mapped, lbl_itos, mean, std


@dataclass
class TaskConfigMultimodal(TaskConfig):
    mainclass:str= "clinical_ts.task.multimodal.MultimodalModel"
    impute_nans:bool = True #impute nans or leave it to the model to handle it
    introduce_nan_columns:bool = False #impute using train set median and introduce an additional column that states if imputation occurred 
    nan_columns_as_cat:bool = False #treat nan columns as categorical variables (instead of continuous)
