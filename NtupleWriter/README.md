Scripts to analyze CMS open data. 

## Make small flat ntuples
Runs on output from [PhysObjectExtractor](https://github.com/cms-opendata-analyses/PhysObjectExtractorTool/tree/master/PhysObjectExtractor). For signal files from delphes, run the scripts in SignalFixer first.

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

## Converting to parquet
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

