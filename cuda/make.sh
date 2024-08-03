export CUDA_HOME=/usr/local/cuda
rm -r build 
python setup.py build 
cp build/lib.linux-x86_64-3.8/CUDA_EXT.cpython-38-x86_64-linux-gnu.so lib/CUDA_EXT.so
