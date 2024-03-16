import ROOT
import pandas as pd 

input_file_name = 'ntuple_SU2L_25_500_v2' ## WITHOUT .root / .parquet
save_file_name = 'ntuple_SU2L_25_500_weighted_v2' ## WITHOUT .root / .parquet

def root_to_df(root_file_path:str):
    df = pd.DataFrame(ROOT.RDataFrame("Events",root_file_path).AsNumpy())
    return df

def df_to_parq(dataframe,name:str):
    return dataframe.to_parquet(name)

def parq_to_df(parquet_file_path:str):
    return pd.read_parquet(parquet_file_path)

## Converting .root to dataframe

df = root_to_df(f"{input_file_name}.root")

#if (df["run"] == 9999).all():
#    print("SIGNAL! Applying weights... ")
#    columns_to_weight = ["HT","LJet_m_plus_RCJet_m_12","bb_m_for_minDeltaR","deltaRLep2ndClosestBJet","deltaRLepClosestBJet"]
#
#    Lumi = 2.256382381 #1/fb
#
#    sigma = 18.92 #fb
#
#    scaled_events = Lumi * sigma
#
#    weigth = scaled_events/50000
#    
#    for col in columns_to_weight:
#        df[col] = df[col]*weigth

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

