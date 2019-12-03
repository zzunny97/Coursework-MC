#include <cuda.h>
#include <stdio.h>
#include <string.h>

__global__ void CountSort(int*, int*, int, int);

__host__ void counting_sort(int* arr, int size, int max_val)
{
	int block_num = 5;
	int thread_num_per_block = 1000;
	uint64_t histo_size = sizeof(int)*max_val*block_num;
	printf("size: %d\n", size);
	printf("max_val: %d\n", max_val);
	printf("block_num: %d\n", block_num);
	printf("thread_per_block: %d\n", thread_num_per_block);

	int* dhisto;
	//memset(histo, 0, histo_size);
	cudaMalloc(&dhisto, histo_size);
	cudaMemset(dhisto, 0, histo_size);
	//cudaMemcpy(dhisto, histo, histo_size, cudaMemcpyHostToDevice);

	int* darr;
	cudaMalloc(&darr, sizeof(int)*size);
	cudaMemcpy(darr, arr, sizeof(int)*size, cudaMemcpyHostToDevice); 

	printf("countsort start\n");
	CountSort<<<block_num, thread_num_per_block>>>(darr, dhisto, size, max_val);
	printf("countsort end\n");
	
	int* histo = (int*)calloc(max_val, sizeof(int)); 
	cudaMemcpy(histo, dhisto, sizeof(int)*max_val, cudaMemcpyDeviceToHost);
	
	
	/*
	int cnt = 0;
	for(int i=0; i<max_val; i++) {
		cnt += histo[i];
	}
	printf("cnt: %d\n", cnt);
	*/
	
	int idx = 0;
	for(int i=0; i<max_val; i++) {
		for(int j=0; j<histo[i]; j++) {
			arr[idx++] = i;
		}
	}
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

	// intialize parallel count arr - need to be fixed
	/*
	int size_per_block = max_val;
	int bstart = bid * size_per_block;
	size_per_thread = size_per_block / thread_per_block;	
	start = bstart + tid * size_per_thread;
	end = start + size_per_thread;
	//if(bid == 0)
	//	printf("=bid: %d tid: %d size_per_thread: %d, start: %d, end: %d\n", bid, tid, size_per_thread, start, end);
	for(int i=start; i<end; i++) {
		dhisto[i] = 0;
	}*/

	//__syncthreads();

	// update histogram	in each block
	if(size % total_block != 0 && bid == total_block - 1) {
		size_per_block = size / total_block + size % total_block;
		bstart = bid * (size / total_block);
		size_per_thread = size_per_block / thread_per_block;
		start = bstart + tid * size_per_thread;
		if(size_per_block % thread_per_block != 0 && 
				tid == thread_per_block - 1) {
			end = start + size_per_thread + size_per_block % thread_per_block;
		}
		else {
			end = start + size_per_thread;
		}
	}
	else {
		size_per_block = size / total_block;
		bstart = bid * size_per_block;	
		size_per_thread = size_per_block / thread_per_block;
		start = bstart + tid * size_per_thread;
		if(size_per_block % thread_per_block != 0 && tid == thread_per_block - 1) {
			end = start + size_per_thread + size_per_block % thread_per_block;
		}
		else {
			end = start + size_per_thread;
		}
	}
	//printf("bid: %d tid: %d start: %d end: %d\n", bid, tid, start, end);
	// ========= version 1 =====
	/*
	for(int i=start; i<end; i++) {
		atomicAdd(&dhisto[darr[i]*total_block + bid], 1);
		//atomicAdd(&dhisto[darr[i]], 1);
		//atomicAdd(&dhisto[0], 1);
	}*/

	// ========= version 2 =====
	for(uint64_t i=start; i<end; i++) {
		atomicAdd(&dhisto[darr[i] + bid * max_val], 1);
	}
	__syncthreads();

	size_per_block = max_val;
	bstart = bid * size_per_block;
	size_per_thread = size_per_block / thread_per_block;
	start = bstart + tid * size_per_thread;
	end = start + size_per_thread;

	if(bid != 0) {
		for(uint64_t i=start; i<end; i++) {
			atomicAdd(&dhisto[i%max_val], dhisto[i]);
		}
	}


	__syncthreads();


}
