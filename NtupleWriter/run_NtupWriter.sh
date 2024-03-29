# rm ntuple.root

echo "Starting NTUP analysis..."
echo "Compiling everything..."
g++ -g3 -o NtupWriter NtupWriter.cxx `root-config --cflags --libs`  `fastjet-install/bin/fastjet-config --cxxflags --libs --plugins`
echo "Compilation done!"

#Test run
# ./NtupWriter "../Samples/CMSfiles/Electron/myoutput_electron_0.root" "test" -1 
./NtupWriter "../Samples/Signal/SU2L_35_500_flat.root" "SU2L_35_500" -1 true

# # Run over multiple files in a directory
# file_prefix="myoutput_"
# file_dir="../Samples/CMSfiles/Electron/"
# for line in $(ls $file_dir$file_prefix*); do 
#     # echo "$line"
#     tag="${line/$file_dir$file_prefix/}"
#     tag="${tag/.root/}"
#     # echo $tag
#     echo ./NtupWriter \"$line\" \"$tag\" -1; 
#     ./NtupWriter "$line" "$tag" -1; 
# done

