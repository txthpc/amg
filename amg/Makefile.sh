rm -rf *.o

gcc -O3 -c SSS_main.c -lm -fopenmp
nvcc -O3 -c SSS_AMG.c -lm -Xcompiler -fopenmp -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
gcc -O3 -c SSS_matvec.c -fopenmp
gcc -O3 -c SSS_utils.c -fopenmp

gcc -O3 -c Setup/SSS_coarsen.c -fopenmp
nvcc -O3 -c Setup/SSS_SETUP.cu -Xcompiler -fopenmp -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
nvcc -O3 -c Setup/SSS_inter.cu -Xcompiler -fopenmp -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

gcc -O3 -c Solve/SSS_SOLVE.c -fopenmp
nvcc -O3 -c Solve/SSS_cycle.cu -Xcompiler -fopenmp
gcc -O3 -c Solve/SSS_smooth.c -fopenmp
nvcc -O3 -c Solve/SSS_cuda.cu -Xcompiler -fopenmp



nvcc -Xcompiler -fopenmp -arch=sm_75 -o amg SSS_main.o SSS_AMG.o SSS_matvec.o SSS_utils.o SSS_coarsen.o SSS_SETUP.o SSS_inter.o SSS_SOLVE.o SSS_cycle.o SSS_smooth.o SSS_cuda.o -lm -lcudart -L/usr/local/cuda/lib64
