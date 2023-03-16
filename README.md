# cuda_dagger
a cuda kernel funcation demo 

```cmd
git clone https://github.com/JackRipper1888/cuda_dagger.git
cd cuda_dagger
nvcc -o main  main.cu -arch=sm_70 -rdc=true

bench 
nvcc -o reduce reduce.cu -arch=sm_70 -lineinfo
ncu -o reduce --set full --import-source yes reduce
./cc
```