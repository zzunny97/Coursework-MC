#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 2000000
#define MAX_VAL 1000000

extern void counting_sort(int arr[], int, int);

int main()
{
  int array[N];

  for(int i=0;i<N;i++){
      array[i] = rand()%MAX_VAL;
  }

  //printf("arr[%d] = %d\n", N-1, array[N-1]);
  counting_sort(array, N, MAX_VAL);

  for(int i=0;i<N-1;i++){
      if( array[i] > array[i+1]){
          printf("Not sorted\n");
          exit(1);
      }
  }
 
  /*
  printf("==========\n");
  for(int i=0; i<1000; i++) {
	 printf("%d\n", array[i]);
  }
  printf("arr[%d] = %d\n", N-1, array[N-1]);
  */

  printf("Sorted\n");
}
