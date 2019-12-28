#include <cuda.h>
#include <stdio.h>
#include <string.h>

__global__ void CountSort(int*, int*, int, int);

__host__ void counting_sort(int* arr, int size, int max_val)
{
	int block_num = 1000;
	int thread_num_per_block = 1000;
	uint64_t histo_size = sizeof(int)*max_val;
	printf("size: %d\n", size);
	printf("max_val: %d\n", max_val);
	printf("block_num: %d\n", block_num);
	printf("thread_per_block: %d\n", thread_num_per_block);
	printf("histo_size: %ld\n", histo_size);

	printf("start cuda malloc\n");
	int* dhisto;
	//memset(histo, 0, histo_size);
	cudaMalloc(&dhisto, (size_t)(histo_size));
	cudaMemset(dhisto, 0, (size_t)(histo_size));
	//cudaMemcpy(dhisto, histo, histo_size, cudaMemcpyHostToDevice);

	int* darr;
	cudaMalloc(&darr, (size_t)(sizeof(int)*size));
	cudaMemcpy(darr, arr, (size_t)(sizeof(int)*size), cudaMemcpyHostToDevice); 
	printf("end cuda malloc\n");

	printf("countsort start\n");
	CountSort<<<block_num, thread_num_per_block>>>(darr, dhisto, size, max_val);
	printf("countsort end\n");
	
	int* histo = (int*)calloc(max_val, sizeof(int)); 
	cudaMemcpy(histo, dhisto, sizeof(int)*max_val, cudaMemcpyDeviceToHost);
	
	
	int cnt = 0;
	for(int i=0; i<max_val; i++) {
		cnt += histo[i];
	}
	printf("cnt: %d\n", cnt);
	
	printf("update arr\n");

	int idx = 0;
	for(int i=0; i<max_val; i++) {
		for(int j=0; j<histo[i]; j++) {
			arr[idx++] = i;
		}
	}
	printf("return to main func\n");
	cudaFree(dhisto);
	cudaFree(darr);
	free(histo);
}

__global__ void CountSort(int* darr, int* dhisto, int size, int max_val) {

	int thread_per_block = blockDim.x;
	int total_block = gridDim.x;
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	uint64_t size_per_block, bstart, size_per_thread, start, end;

	int myrank = bid * thread_per_block + tid;
	int range_per_thread = max_val / (thread_per_block * total_block);
	int ts = myrank * range_per_thread;
	int te = ts + range_per_thread;
	for(int i=0; i<size; i++) {
		if(darr[i] >= ts && darr[i] < te) {
			dhisto[darr[i]]++;
			//atomicAdd(&dhisto[darr[i]], 1);
		}
	}

	__syncthreads();



}
