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
skiplist<int, int> list(0,1000000);

pthread_mutex_t m1=PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m2=PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m3=PTHREAD_MUTEX_INITIALIZER;

typedef struct padded_int {
	long local_odd;
	long local_sum;
	char padding[48];
} private_cnt;


class Node{
public:
	char action;		//1
	int num;				//4
	Node* next;			//8
	char padding[3];
	Node(char act, int n);
};

Node::Node(char act, int n) {
	action = act;
	num = n;
	padding[0]=0;
	padding[1]=0;
	padding[2]=0;
	next = NULL;
}

class Queue{
public:
	int size;
	Node* front;
	Node* rear;
	
	Queue();
	void enq(char act, int n);
	Node* deq();
	void print();
};

Queue::Queue() {
	size = 0;
	front = NULL;
	rear = NULL;
}

void Queue::enq(char act, int n) {
	Node* insert = new Node(act, n);
	if(size==0) {
		front = insert;
		rear = insert;
	} else{
		rear->next = insert;	
		rear = insert;
	}
	size++;
}

Node* Queue::deq() {
	Node* ret = front;
	front = front->next;
	return ret;
}

void Queue::print() {
	Node* cur = front;
	while(cur!=NULL) {
		cout << cur->action << " " << cur->num << endl;
		cur = cur->next;
	}
}

void* do_work(void* tid);
Queue q;
int idx;

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

	FILE* fin;
	fin = fopen(fn, "r");
	char action;
	long num;
	int line=0;
	while (fscanf(fin, "%c %ld\n", &action, &num) == 2) {
		q.enq(action, num);
	}

	
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

	fclose(fin);
	clock_gettime( CLOCK_REALTIME, &stop);

	cout << list.printList() << endl;
	cout << sum << " " << odd << endl;
	cout << "Elapsed time: " << (stop.tv_sec - start.tv_sec) + ((double) (stop.tv_nsec - start.tv_nsec))/BILLION << " sec" << endl;
	pthread_attr_destroy(&attr);
	pthread_exit(NULL);
	// clean up and return
	return (EXIT_SUCCESS);

}

void* do_work(void* tid)
{
	struct timespec start, stop;
	struct timespec tmp1, tmp2;
	double i_overhead=0, q_overhead=0, w_overhead=0;
	clock_gettime( CLOCK_REALTIME, &start);


	char action;
	long num;
	private_cnt pcnt;
	while (1) {
		pthread_mutex_lock(&m1);    // how about spinlock here??
		if(idx==q.size) {
			pthread_mutex_unlock(&m1);
			break;
		}
		Node* inst = q.deq();
		idx++;
        if(idx % 10000 == 0) cout << idx << endl;
		
		pthread_mutex_unlock(&m1);
		action = inst->action;
		num = inst->num;
		//cout << "TID: " << *(int*)tid << " " << action << " " << num << endl;
		//pthread_mutex_unlock(&mutex);
		//cout << *(int*)tid << " " << action << " " << num << endl;
		if (action == 'i') {            // insert
			clock_gettime(CLOCK_REALTIME, &tmp1);
			list.insert(num,num);
			pcnt.local_sum += num;
			if (num % 2 == 1) {
				pcnt.local_odd++;
			}
			clock_gettime(CLOCK_REALTIME, &tmp2);
			i_overhead += (tmp2.tv_sec - tmp1.tv_sec) + ((double) (tmp2.tv_nsec-tmp1.tv_nsec))/BILLION;
		}else if (action == 'q') {      // qeury
			clock_gettime(CLOCK_REALTIME, &tmp1);
			if(list.find(num)!=num)
				cout << "ERROR: Not Found: " << num << endl;
			clock_gettime(CLOCK_REALTIME, &tmp2);
			q_overhead += (tmp2.tv_sec - tmp1.tv_sec) + ((double) (tmp2.tv_nsec-tmp1.tv_nsec))/BILLION;
		} else if (action == 'w') {     // wait
			clock_gettime(CLOCK_REALTIME, &tmp1);
			usleep(num*1000);
			clock_gettime(CLOCK_REALTIME, &tmp2);
			w_overhead += (tmp2.tv_sec - tmp1.tv_sec) + ((double) (tmp2.tv_nsec-tmp1.tv_nsec))/BILLION;
		} else {
			printf("ERROR: Unrecognized action: '%c'\n", action);
			pthread_exit(0);
		}
	}

	clock_gettime( CLOCK_REALTIME, &stop);
	pthread_mutex_lock(&m2);
	cout << "TID: " << *(int*)tid << " Elapsed time: " << (stop.tv_sec - start.tv_sec) + ((double) (stop.tv_nsec - start.tv_nsec))/BILLION << " sec" << endl;
	cout << "i_overhead = " << i_overhead << endl;
	cout << "q_overhead = " << q_overhead << endl;
	cout << "w_overhead = " << w_overhead << endl;
	
	odd += pcnt.local_odd;
	sum += pcnt.local_sum;
	pthread_mutex_unlock(&m2);

}

