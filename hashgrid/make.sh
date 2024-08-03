export CUDA_HOME=/usr/local/cuda
rm -rf build 
python setup.py build 
cp build/lib.linux-x86_64-3.8/HASHGRID.cpython-38-x86_64-linux-gnu.so lib/HASHGRID.so