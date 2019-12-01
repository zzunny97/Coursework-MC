#include <cuda.h>
#include <stdio.h>

__global__ void CountSort(int*, int*, int, int);

__host__ void counting_sort(int arr[], int size, int max_val)
{
	int block_num = 2;
	int thread_num_per_block = 5;

	int* dhisto;
	int* histo = (int*)malloc(sizeof(int)*max_val*block_num);
	memset(histo, 0, sizeof(int)*max_val*block_num);
	cudaMalloc(&dhisto, sizeof(int)*max_val*block_num);
	cudaMemcpy(dhisto, histo, sizeof(int)*max_val*block_num, cudaMemcpyHostToDevice);
	printf("dhisto init complete\n");

	// init device data arr
	int* darr;
	cudaMalloc(&darr, sizeof(int)*size);
	cudaMemcpy(darr, arr, sizeof(int)*size, cudaMemcpyHostToDevice); 
	printf("darr init complete\n");

	printf("func: CountSort start\n");
	CountSort<<<block_num, thread_num_per_block, sizeof(int)*max_val>>>(darr, dhisto, size, max_val);
	printf("func: CountSort end\n");


	cudaMemcpy(histo, dhisto, sizeof(int)*max_val*block_num, cudaMemcpyDeviceToHost);

	int cnt = 0;
	for(int i=0; i<max_val*block_num; i++) {
		cnt += histo[i];
	}
	printf("cnt: %d\n", cnt);
	
	int idx = 0;
	int* histo2 = (int*)malloc(sizeof(int)*max_val);
	memset(histo2, 0, sizeof(int)*max_val);
	for(int i=0; i<max_val; i++) {
		for(int j=0; j<2; j++) {
			histo2[i] += histo[i*2+j];
		}
	}
	int cnt2=0;
	for(int i=0; i<max_val; i++) {
		cnt2 += histo2[i];
	}
	printf("cnt2: %d\n", cnt2);

	for(int i=0; i<max_val; i++) {
		for(int j=0; j<histo2[i]; j++) {
			arr[idx++] = i;
		}
	}
	//cudaMemcpy(arr, darr, sizeof(int)*size, cudaMemcpyDeviceToHost);

	cudaFree(dhisto);
	cudaFree(darr);
	free(histo);
}

__global__ void CountSort(int* darr, int* dhisto, int size, int max_val) {

	/*
	if(blockIdx.x == 0 && threadIdx.x ==0 ) {
		printf("blockIdx.x: %d\n", blockIdx.x);
		printf("threadIdx.x: %d\n", threadIdx.x);
		printf("blockDim.x: %d\n", blockDim.x);
		printf("gridDim.x: %d\n", gridDim.x);
	}*/
	int thread_per_block = blockDim.x;
	int total_block = gridDim.x;
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int size_per_thread, start, end;

	extern __shared__ int count[];

	// intialize parallel count arr
	size_per_thread = max_val / thread_per_block;	
	start = tid * size_per_thread;
	end = start + size_per_thread;
	printf("tid: %d size_per_thread: %d, start: %d, end: %d\n", tid, size_per_thread, start, end);
	//for(int i=start; i<end; i++)
	//	count[i] = 0;
	memset(&count[start], 0, sizeof(int)*size_per_thread);
	__syncthreads();



	// update histogram	in each block
	int size_per_block = size / total_block;
	int bstart = bid * size_per_block;	
	size_per_thread = size_per_block / thread_per_block;
	start = bstart + tid * size_per_thread;
	end = start + size_per_thread;
	printf("tid: %d size_per_thread: %d, start: %d, end: %d\n", tid, size_per_thread, start, end);
	for(int i=start; i<end; i++) {
		atomicAdd(&count[darr[i]], 1);
	}
	__syncthreads();

	size_per_thread = max_val / thread_per_block;
	start = tid * size_per_thread;
	end = start + size_per_thread;
	printf("tid: %d size_per_thread: %d, start: %d, end: %d\n", tid, size_per_thread, start, end);
	for(int i=start; i<end; i++) {
		//if(bid == 0 && tid == 1)
		//	printf("idx: %d\n", total_block*i + bid);
		dhisto[total_block * i + bid] = count[i];
	}
	__syncthreads();
}
