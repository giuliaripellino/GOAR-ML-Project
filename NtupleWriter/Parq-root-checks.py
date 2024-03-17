import ROOT
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv

input_file_name = 'ntuple_SU2L_25_500_v2' ## WITHOUT .root / .parquet
save_file_name = 'ntuple_SU2L_25_500_weighted_v2' ## WITHOUT .root / .parquet

"""
Some necessary functions are first defined
"""

def root_to_df(root_file_path:str):
    df = pd.DataFrame(ROOT.RDataFrame("Events",root_file_path).AsNumpy())
    return df

def df_to_parq(dataframe,name:str):
    return dataframe.to_parquet(name)

def parq_to_df(parquet_file_path:str):
    return pd.read_parquet(parquet_file_path)

def MinMaxScaling(df,variables,file_name):
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df[variables])

    scaling_factors = {}
    for index,variable in enumerate(variables):
        scaling_factors[variable] = {'min': scaler.data_min_[index], 'max':scaler.data_max_[index]}

    with open(file_name,mode='w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Variable','Min','Max'])
        for Variable,factors in scaling_factors.items():
            writer.writerow([Variable, factors['min'],factors['max']])
    return scaled_df

def UnScaling(df,scaling_factors_path,variables):
    scaling_factors = {}
    with open(scaling_factors_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            feature = row['Variable']
            min_val = float(row['Min'])
            max_val = float(row['Max'])
            scaling_factors[feature] = {'min': min_val, 'max': max_val}
    unscaled = df
    for variable in variables:
        min_val = scaling_factors[variable]['min']
        max_val = scaling_factors[variable]['max']

        unscaled[variable] = df[variable] * (max_val - min_val) + min_val

    return unscaled

"""
Here comes the actual conversion with tests
"""
variables = ["HT","LJet_m_plus_RCJet_m_12","bb_m_for_minDeltaR","deltaRLep2ndClosestBJet","deltaRLepClosestBJet"]

## Converting .root to dataframe
df = root_to_df(f"{input_file_name}.root")

## MinMaxScaling the wanted variables and saving the scaling factors
df[variables] = MinMaxScaling(df,variables,'scaling_factors.csv')


## Saving the dataframe as parquet
df.to_parquet(f"{save_file_name}.parquet")

## converting parquet to dataframe
df_from_parq = parq_to_df(f"{save_file_name}.parquet")

## Checking if the made parquet and the original .root are equal
diff = df_from_parq - df
for col in diff.columns:
    if (diff[col] == 0).all():
        print(f"All values in '{col}' are zero. Good!")
    else:
        print(f"'{col}' has non-zero differences. Something is wrong!")

### To unscale, do something like this: unscaled_Dataframe = UnScaling()

"""
# Old function not necessary

if (df["run"] == 9999).all():
   print("SIGNAL! Applying weights... ")
   columns_to_weight = variables

   Lumi = 2.256382381 #1/fb

   sigma = 18.92 #fb

   scaled_events = Lumi * sigma

   weigth = scaled_events/50000
   
   for col in columns_to_weight:
       df[col] = df[col]*weigth

"""