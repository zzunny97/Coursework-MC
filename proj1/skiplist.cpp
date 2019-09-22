/*
 * main.cpp
 *
 * Serial version
 *
 * Compile with -O2
 */

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include "skiplist2.h"

using namespace std;


// aggregate variables
long sum = 0;
long odd = 0;
long min = INT_MAX;
long max = INT_MIN;
bool done = false;

int num_threads;
FILE* fin;
skiplist<int, int> list(0,1000000);

pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;


// function prototypes

void* do_work(void* tid)
{
	char action;
	long num;
	while (fscanf(fin, "%c %ld\n", &action, &num) == 2) {
		pthread_mutex_lock(&mutex);
		cout << "tid = " << *(int*)tid << " " << action << " " << num << endl;
		pthread_mutex_unlock(&mutex);
		if (action == 'i') {            // insert
			list.insert(num,num);
			// update aggregate variables
			sum += num;
			if (num % 2 == 1) {
				odd++;
			}
		}else if (action == 'q') {      // qeury
			if(list.find(num)!=num)
				cout << "ERROR: Not Found: " << num << endl;
		} else if (action == 'w') {     // wait
			usleep(num*1000);
		} else {
			printf("ERROR: Unrecognized action: '%c'\n", action);
			pthread_exit(0);
		}
	}


}



int main(int argc, char* argv[])
{
	struct timespec start, stop;


	// check and parse command line options
	if (argc != 3) {
		printf("Usage: sum <infile> <num_threads>\n");
		exit(EXIT_FAILURE);
	}
	clock_gettime( CLOCK_REALTIME, &start);

	char *fn = argv[1];
	num_threads = atoi(argv[2]);
	fin = fopen(fn, "r");
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	pthread_t thread[num_threads];
	int tid[num_threads] = {0,};

	for(int i=0; i<num_threads; i++) {
		tid[i] = i;
		pthread_create(&thread[i], NULL, do_work, (void*)&tid[i]);
	}

	for(int i=0; i<num_threads; i++) {
		pthread_join(thread[i], NULL);
	}

	// load numbers and add them to the queue
	fclose(fin);
	clock_gettime( CLOCK_REALTIME, &stop);

	// print results
	cout << list.printList() << endl;
	cout << sum << " " << odd << endl;
	cout << "Elapsed time: " << (stop.tv_sec - start.tv_sec) + ((double) (stop.tv_nsec - start.tv_nsec))/BILLION << " sec" << endl;
	pthread_attr_destroy(&attr);
	pthread_exit(NULL);
	// clean up and return
	return (EXIT_SUCCESS);

}

