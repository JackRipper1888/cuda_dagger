# cuda_dagger
a cuda kernel funcation demo 
```cmd
git clone https://github.com/JackRipper1888/cuda_dagger.git
cd cuda_dagger
nvcc -o cc  main.cu -arch=sm_70 -rdc=true
./cc
```