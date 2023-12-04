cd build
cmake -DCMAKE_PREFIX_PATH=F:/Programming/libtorch-win-shared-with-deps-2.0.1+cu117 ..
cmake --build . --config Release
cd Release
emotion-classifier.exe "i can't be sad. unaffable"
pause