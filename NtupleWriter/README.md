Scripts to analyze CMS open data. 

## Make small flat ntuples
Runs on output from [PhysObjectExtractor](https://github.com/cms-opendata-analyses/PhysObjectExtractorTool/tree/master/PhysObjectExtractor).

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


