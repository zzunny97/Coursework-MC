#include <cuda.h>
#include <stdio.h>

__global__ void CountSort(int*, int*, int, int);

__host__ void counting_sort(int arr[], int size, int max_val)
{
	int block_num = 100;
	int thread_num_per_block = 50;
	uint64_t histo_size = sizeof(int)*max_val*block_num;
	printf("size: %d\n", size);
	printf("max_val: %d\n", max_val);
	printf("block_num: %d\n", block_num);
	printf("thread_per_block: %d\n", thread_num_per_block);

	int* dhisto;
	int* histo = (int*)malloc(histo_size);
	memset(histo, 0, histo_size);
	cudaMalloc(&dhisto, histo_size);
	cudaMemcpy(dhisto, histo, histo_size, cudaMemcpyHostToDevice);

	int* darr;
	cudaMalloc(&darr, sizeof(int)*size);
	cudaMemcpy(darr, arr, sizeof(int)*size, cudaMemcpyHostToDevice); 

	CountSort<<<block_num, thread_num_per_block>>>(darr, dhisto, size, max_val);
	cudaMemcpy(histo, dhisto, histo_size, cudaMemcpyDeviceToHost);

	/*
	int cnt = 0;
	for(int i=0; i<max_val*block_num; i++) {
		cnt += histo[i];
	}

	for(int i=0; i<max_val*block_num; i++) {
		if(histo[i] != 0) {
			printf("%d: %d\n", i % max_val, histo[i]);
		}
	}
	printf("cnt: %d\n", cnt);
	*/
	
	int idx = 0;
	int* histo2 = (int*)malloc(sizeof(int)*max_val);

	memset(histo2, 0, sizeof(int)*max_val);

	// ======== version 1 ======
	/*
	for(int i=0; i<max_val; i++) {
		for(int j=0; j<block_num; j++) {
			histo2[i] += histo[i*block_num+j];
		}
	}
	*/

	// ======== version 2 ======
	for(int i=0; i<block_num; i++) {
		for(int j=0; j<max_val; j++) {
			histo2[j] += histo[i*max_val+j];
		}
	}

	/*
	int cnt2=0;
	for(int i=0; i<max_val; i++) {
		cnt2 += histo2[i];
	}
	printf("cnt2: %d\n", cnt2);
	*/

	for(int i=0; i<max_val; i++) {
		for(int j=0; j<histo2[i]; j++) {
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
	int size_per_block, bstart, size_per_thread, start, end;

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

	__syncthreads();

	// update histogram	in each block
	size_per_block = size / total_block;
	bstart = bid * size_per_block;	
	size_per_thread = size_per_block / thread_per_block;
	start = bstart + tid * size_per_thread;
	end = start + size_per_thread;
	// ========= version 1 =====
	/*
	for(int i=start; i<end; i++) {
		atomicAdd(&dhisto[darr[i]*total_block + bid], 1);
		//atomicAdd(&dhisto[darr[i]], 1);
		//atomicAdd(&dhisto[0], 1);
	}*/

	// ========= version 2 =====
	for(int i=start; i<end; i++) {
		atomicAdd(&dhisto[darr[i] + bid * max_val], 1);
	}
	__syncthreads();

}
