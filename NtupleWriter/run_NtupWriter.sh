rm ntuple.root
# root -l 'NtupWriter.cxx(-1, "../myoutput_merge.root")'

echo "Starting NTUP analysis..."
echo "Compiling everything..."
g++ -g3 -o NtupWriter NtupWriter.cxx `root-config --cflags --libs`  `fastjet-install/bin/fastjet-config --cxxflags --libs --plugins`
echo "Compilation done!"

./NtupWriter "../myoutput_electron_merge.root" -1