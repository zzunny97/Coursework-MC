#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define N 100000000
#define MAX_VAL 100000000


int main()
{
  int *array = (int*)malloc(sizeof(int)*N);
  int *histo = (int*)malloc(sizeof(int)*MAX_VAL);

  for(int i=0;i<N;i++){
      array[i] = rand()%MAX_VAL;
  }

  memset(histo, 0, sizeof(int)*MAX_VAL);
  //for(int i=0; i<MAX_VAL; i++)
  //	  histo[i] = 0;

  for(int i=0; i<N; i++)
	  histo[array[i]]++;

  int idx = 0;
  for(int i=0; i<MAX_VAL; i++) {
	 for(int j=0; j<histo[i]; j++) {
		array[idx++] = i;
	 }
  }

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
