mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=F:/Programming/libtorch-win-shared-with-deps-2.0.1+cu117 ..
cmake --build . --config Release
pause