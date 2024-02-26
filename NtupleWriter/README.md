Scripts to analyze CMS open data. 

## Make small flat ntuples
Runs on output from [PhysObjectExtractor](https://github.com/cms-opendata-analyses/PhysObjectExtractorTool/tree/master/PhysObjectExtractor). For signal files from delphes, run the fixupSignal script first.

### Setup

Needs FastJet for reclustering.

Fetch FastJet:
```
curl -O http://fastjet.fr/repo/fastjet-3.4.2.tar.gz 
tar zxvf fastjet-3.4.2.tar.gz
cd fastjet-3.4.2/
```

Compile FastJet:
```
./configure --prefix=$PWD/../fastjet-install
make 
make check
make install
cd ..
```

### Compile and Run
```
source run_NtupWriter.sh
```

## Plotting
```
python plot_ControlHistos.py
```
```
python plot_SRVariables.py
```

## Handling signal files 
Signal files from Delphes can be converted to the same format as the CMS open data.

```
git clone https://github.com/delphes/delphes.git
cd delphes
make
root -l ../fixupSignal.cxx
```

## Converting to parquet
### 1.
```
pyspark
import ROOT
import pandas as pd
root_file = ROOT.TFile.Open("Input/ntuple_em.root")
tree = root_file.Get("Events")
dataframe = pd.DataFrame(tree.AsMatrix())
leaves = tree.GetListOfLeaves()
leaf_names = [leaf.GetName() for leaf in leaves]
dataframe.columns = leaf_names
dataframe.to_parquet("ntuple_em.parquet")
```

### 2. 
One can run ```Parq-root-checks.py``` and give the .root file as input (top of the python file), which gives an .parquet file. If the input is signal (ATLAS convention where run=9999 for generated signal) **Is this correct?**, the signal is scaled. This script also checks that the created .parquet and the original .root file equal. 

