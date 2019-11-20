#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <cmath>

int N;
typedef double Data;
using namespace std;

void Print_mat(Data** p, int size) {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            //cout << p[i][j] << "\t";
            printf("%.2f\t", p[i][j]);
        }
        cout << endl;
    }
    cout << endl;
}

void Copy_mat(Data** src, Data** dst) {
    for(int i=0; i<N; i++) 
        memcpy(dst[i], src[i], sizeof(Data)* N);
}

void Free_mat(Data** p, int size) {
    for(int i=0; i<size; i++)
        delete[] p[i];
    delete[] p;
}

void Inverse_mat(Data** p, int size) {
    //cout << "inverse start" << endl;
    if(size < 2)
        return;
    else if(size == 2) {
        Data prefix = p[0][0]*p[1][1] - p[0][1]*p[1][0];
        for(int i=0; i<size; i++) {
            for(int j=0; j<size; j++) {
                p[i][j] *= prefix;
            }
        }
        Data tmp = p[1][1];
        p[1][1] = p[0][0];
        p[0][0] = tmp;
        p[0][1] *= -1;
        p[1][0] *= -1;
        return;
        
    }
    Data** mat = new Data*[size];
    for(int i=0; i<size; i++)
        mat[i] = new Data[2*size];

    //cout << "zzunny1" << endl;
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            mat[i][j] = p[i][j];
        }
    }
    //cout << "zzunny2" << endl;
    for(int i=0; i<size; i++) {
        for(int j=size; j< 2*size; j++) {
            if(j==(i+size)) mat[i][j] = 1;
        }
    }
    
	/*
    for(int i=0; i<size; i++) {
        for(int j=0; j<2*size; j++) {
            cout << mat[i][j] << "\t";
        }
        cout << endl;
    }*/
    
    
    //cout << "zzunny3" << endl;
    Data tmp;
    
    for(int i=size-1; i>1; i--) {
        //cout << "zzunny3-1" << endl;
        if(mat[i-1][1] < mat[i][1]) {
            //cout << "zzunny3-2" << endl;
            for(int j=0; j<2*size; j++) {
                tmp = mat[i][j];
                mat[i][j] = mat[i-1][j];
                mat[i-1][j] = tmp;
            }
        }
    }

    //cout << "zzunny4" << endl;
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            if(j!=i) {
                tmp = mat[j][i] / mat[i][i];
                for(int k=0; k<size*2; k++)
                    mat[j][k] -= mat[i][k]*tmp;
            }
        }
    }
    //cout << "zzunny5" << endl;
    Data** ret = new Data*[size];
    for(int i=0; i<size; i++)
        ret[i] = new Data[size];

    for(int i=0; i<size; i++) 
        for(int j=0; j<size; j++) 
            ret[i][j] = mat[i][j];
    //cout << "inverse end" << endl;
}

void verify(Data** A, Data** L, Data** U, int size) {
    double diffMax = 0.0;
    Data** C = new Data*[size];
    for(int i=0; i<size; i++) {
        C[i] = new Data[size];
        memset(C[i], 0, sizeof(Data) * size);
    }
    for(int i=0; i<size; i++)
        for(int j=0; j<size; j++)
            for(int k=0; k<size; k++) 
                C[i][j] += L[i][k] * U[k][j];

    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            double target = A[i][j] - C[i][j];
            if(target > diffMax)
                diffMax = target;
        }
    }
    printf("diffMax: %.5f\n", diffMax);
    for(int i=0; i<size; i++)
        delete[] C[i];
    delete[] C;
}

void Mul_mat(Data** a, Data** b, Data** ret, int size) {
    for(int i=0; i<size; i++) {
        memset(ret[i], 0, sizeof(Data) * size);
    }
    for(int i=0; i<size; i++)
        for(int j=0; j<size; j++)
            for(int k=0; k<size; k++) 
                ret[i][j] += a[i][k] * b[k][j];
}

void Sub_mat(Data** a, Data** b, int size) {
	for(int i=0; i<size; i++)
		for(int j=0; j<size; j++)
			a[i][j] -= b[i][j];
}


void block_LU(Data** a, Data** l, Data** u, int size) {
    for(int k=0; k<size; k++) {
        u[k][k] = a[k][k];
        for(int i=k+1; i<size; i++) {
            l[i][k] = a[i][k] / u[k][k];
            u[k][i] = a[k][i];
        }
        for(int i=k+1; i<size; i++)
            for(int j=k+1; j<size; j++)
                a[i][j] -= l[i][k] * u[k][j];
    }
}

void Copy_to_one(Data** src, Data* dst, int size) {
	for(int i=0; i<size; i++)
		for(int j=0; j<size; j++)
			dst[i*size+j] = src[i][j];
}

void Copy_to_two(Data* src, Data** dst, int size) {
	for(int i=0; i<size; i++)
		for(int j=0; j<size; j++)
			dst[i][j] = src[i*size+j];
}


void LU(Data** A, Data** L, Data** U) {
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if N = 15 and size = 9
    int block_sq = sqrt(size);      // 3
    int row_per_block = (N / block_sq);     // 5
    int col_per_block = (N / block_sq);     // 5
    //cout << "bound: " << bound << endl;
    int x_start = (rank / block_sq) * row_per_block;    // start of x in the process
    int x_end = x_start + row_per_block;                // end of x in the process
    int y_start = (rank % block_sq) * row_per_block;    // start of y in the process
    int y_end = y_start + row_per_block;                // end of y in the process
	long oned_size = pow(row_per_block, 2);             // number of elements of a block
	long idx = 0;
    cout << "rank: "  << rank <<  " x_start: " << x_start << " x_end: " << x_end
        << " y_start: " << y_start << " y_end: " << y_end << endl;
   
    // ====== INITIALIZE SMALL MATRICES ==== 
    Data** small_A = new Data*[row_per_block];          // block A
    Data** small_L = new Data*[row_per_block];          // block L
    Data** small_U = new Data*[row_per_block];          // block U
	Data* oned_mat = new Data[oned_size];               // oned_size == pow(row_per_block ,2)
	Data* oned_mat2 = new Data[oned_size];
    for(int i=0; i<row_per_block; i++) {
        small_A[i] = new Data[row_per_block];
        small_L[i] = new Data[row_per_block];
        small_U[i] = new Data[row_per_block];
    }

    for(int i=0; i<row_per_block; i++) small_L[i][i] = 1;
    
    for(int i=x_start; i<x_end; i++)
        for(int j=y_start; j<y_end; j++)
            small_A[i-x_start][j-y_start] = A[i][j];

    // if the process is master
	if(rank == 0) {
		block_LU(small_A, small_L, small_U, row_per_block);	
		for(int i=0; i<row_per_block; i++)
			for(int j=0; j<row_per_block; j++)
				L[i][j] = small_L[i][j];
		for(int i=0; i<row_per_block; i++)
			for(int j=0; j<row_per_block; j++)
				U[i][j] = small_U[i][j];

		// send to row
        for(int dst = 1; dst<block_sq; dst++) {
            Inverse_mat(small_L, row_per_block);
			Copy_to_one(small_L, oned_mat, row_per_block); 
			MPI_Send(&oned_mat[0], oned_size, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD); // send to row the inverse of L
        }
        cout << "Rank 0: send to row complete" << endl;

        // send to col
        for(int dst = block_sq; dst<size; dst+=block_sq) {
            Inverse_mat(small_U, row_per_block);
			Copy_to_one(small_U, oned_mat, row_per_block);
			MPI_Send(&oned_mat[0], oned_size, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD); // send to col the inverse of U
        }
        cout << "Rank 0: send to col complete" << endl;

		// gather all from other processes
		for(int i=0; i<block_sq; i++) {
			for(int j=0; j<block_sq; j++) {
				int src = i*block_sq + j; 
				x_start = (src / block_sq) * row_per_block;
				y_start = (src % block_sq) * row_per_block;
				// get both
				if(i==j) {
					if(i==0 && j==0) continue;
                    cout << "rank 0: waiting for rank-" << src << endl;
					MPI_Recv(&oned_mat[0], oned_size, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	// recv L
					MPI_Recv(&oned_mat2[0], oned_size, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // recv U
					idx = 0;
					for(int i=x_start; i<x_start+row_per_block; i++) 
						for(int j=y_start; j<y_start+row_per_block; j++)
							L[i][j] = oned_mat[idx++];
					idx = 0;
					for(int i=x_start; i<x_start+row_per_block; i++) 
						for(int j=y_start; j<y_start+row_per_block; j++)
							U[i][j] = oned_mat2[idx++];
                    cout << "rank 0: gather from rank-" << src << endl;
				}
				// get only L
				else if(i > j) {
                    cout << "rank 0: waiting for rank-" << src << endl;
					MPI_Recv(&oned_mat[0], oned_size, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv L
					idx = 0;
					for(int i=x_start; i<x_start+row_per_block; i++) 
						for(int j=y_start; j<y_start+row_per_block; j++)
							L[i][j] = oned_mat[idx++];
                    cout << "rank 0: gather from rank-" << src << endl;
				}
				// get only U
				else {
                    cout << "rank 0: waiting for rank-" << src << endl;
					MPI_Recv(&oned_mat2[0], oned_size, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv L
					idx = 0;
					for(int i=x_start; i<x_start+row_per_block; i++) 
						for(int j=y_start; j<y_start+row_per_block; j++)
							U[i][j] = oned_mat[idx++];
                    cout << "rank 0: gather from rank-" << src << endl;
				}
			}
		}
	}

	// center processes 
	else if(x_start == y_start) {
		// initialize mul mat
		Data** mul = new Data*[row_per_block];
		for(int i=0; i<row_per_block; i++)
			mul[i] = new Data[row_per_block];

		// recv and subtract
		for(int i=0; i<rank/block_sq; i++) {
			MPI_Recv(&oned_mat[0], oned_size, MPI_DOUBLE, ((rank % block_sq) * block_sq) + i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    // receive L
			MPI_Recv(&oned_mat2[0], oned_size, MPI_DOUBLE, rank % block_sq + (i*block_sq), 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);       // receive U
			idx = 0;
			for(int i=0; i<row_per_block; i++)
				for(int j=0; j<row_per_block; j++)
					small_L[i][j] = oned_mat[idx++];
			idx=0;
			for(int i=0; i<row_per_block; i++)
				for(int j=0; j<row_per_block; j++)
					small_U[i][j] = oned_mat2[idx++];
			// multiply two matrices from row and col, and subtract from original A
			Mul_mat(small_L, small_U, mul, row_per_block);
			Sub_mat(small_A, mul, row_per_block);	
		}

		// block LU
		block_LU(small_A, small_L, small_U, row_per_block);
			
		// send to master both L and U
		Copy_to_one(small_L, oned_mat, row_per_block);
		Copy_to_one(small_U, oned_mat2, row_per_block);
		MPI_Send(&oned_mat[0], oned_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);	// send final 'L' to master
		MPI_Send(&oned_mat2[0], oned_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);  // send final 'U' to master

		if(rank != size - 1) {
			// send to row the inverse of L and U
			Inverse_mat(small_L, row_per_block);
            Inverse_mat(small_U, row_per_block);
			int limit = (rank/block_sq+1) * block_sq;
			for(int dst = rank+1; dst<limit; dst++) {
				Copy_to_one(small_L, oned_mat, row_per_block); 
				MPI_Send(&oned_mat[0], oned_size, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD); // send row the inverse of L
			}

			// send to col the inverse of U
			for(int dst = block_sq; dst<size; dst+=block_sq) {
				Copy_to_one(small_U, oned_mat2, row_per_block);
				MPI_Send(&oned_mat2[0], oned_size, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD); // send col the inverse of U
			}
		}
		Free_mat(mul, row_per_block);
	}

	// rest rank != 0 and !center
	else {
		// initialize mul mat
		Data** mul = new Data*[row_per_block];
		for(int i=0; i<row_per_block; i++)
			mul[i] = new Data[row_per_block];

        int iter = x_start > y_start? rank%block_sq : rank/block_sq;
        // multiply two matrices from row and col, and subtract from original A
        for(int i=0; i<iter; i++) {
            MPI_Recv(&oned_mat[0], oned_size, MPI_DOUBLE, i*block_sq + rank % block_sq, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // recv L
            MPI_Recv(&oned_mat2[0], oned_size, MPI_DOUBLE, (rank/block_sq)*block_sq + i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv U
            idx = 0;
            for(int i=0; i<row_per_block; i++)
                for(int j=0; j<row_per_block; j++)
                    small_L[i][j] = oned_mat[idx++];
            idx=0;
            for(int i=0; i<row_per_block; i++)
                for(int j=0; j<row_per_block; j++)
                    small_U[i][j] = oned_mat2[idx++];

            Mul_mat(small_L, small_U, mul, row_per_block);
            Sub_mat(small_A, mul, row_per_block);   
        }
		if(x_start > y_start) {
            // Recv inverse of U from center process
			MPI_Recv(&oned_mat[0], oned_size, MPI_DOUBLE, (rank / block_sq) * block_sq + ( rank % block_sq ), 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv inverse U
			idx = 0;			
			for(int i=0; i<row_per_block; i++)
				for(int j=0; j<row_per_block; j++)
					small_U[i][j] = oned_mat[idx++];
            // get final L and now it's time to send to master
			Mul_mat(small_A, small_U, mul, row_per_block);
            // send final L to master
            Copy_to_one(mul, oned_mat, row_per_block);
            MPI_Send(&oned_mat[0], oned_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            // send final L to row
            for (int dst = rank+1; dst < (rank / block_sq) * block_sq + (block_sq-1); dst++) {
                MPI_Send(&oned_mat[0], oned_size, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
            }
		}
		else if (x_start < y_start) {
            // Recv inverse of L from center process
			MPI_Recv(&oned_mat[0], oned_size, MPI_DOUBLE, (rank / block_sq) * block_sq + ( rank % block_sq ), 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv inverse L
			idx = 0;			
			for(int i=0; i<row_per_block; i++)
				for(int j=0; j<row_per_block; j++)
					small_L[i][j] = oned_mat[idx++];
            // get final U and now it's time to send to master
			Mul_mat(small_A, small_L, mul, row_per_block); 
            // send final U to master
            Copy_to_one(mul, oned_mat, row_per_block);
            MPI_Send(&oned_mat[0], oned_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            // send final U to col
            for(int dst = rank + block_sq; dst<size; dst+=block_sq) {
                MPI_Send(&oned_mat[0], oned_size, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
            }
		}
	}
		


	if(rank == 0) {
		cout << "===== L =====" << endl;
		Print_mat(L, N);
		cout << "===== U =====" << endl;
		Print_mat(U, N);

	}
    MPI_Finalize();
}

int main(int argc, char** argv){
    if(argc != 3 ) {
        cout << "Usage: ./proj3 N Seed" << endl;
        exit(-1);
    }
    N = atoi(argv[1]);
    int seed = atoi(argv[2]);
    srand(seed);

    //cout << "Initialize" << endl;
    Data** A_original = new Data*[N];    
    for(int i=0; i<N; i++) A_original[i] = new Data[N];

    Data** A = new Data*[N];
    for(int i=0; i<N; i++) {
        A[i] = new Data[N];
        memset(A[i], 0, sizeof(Data) * N);
    }

    Data** L = new Data*[N];
    for(int i=0; i<N; i++) {
        L[i] = new Data[N];
        memset(L[i], 0, sizeof(Data) * N);
    }

    Data** U = new Data*[N];
    for(int i=0; i<N; i++) {
        U[i] = new Data[N];
        memset(U[i], 0, sizeof(Data) * N);
    }

    for(int i=0; i<N; i++) L[i][i] = 1;
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            A[i][j] = rand() % 10 + 1;

    Copy_mat(A, A_original);
    //Print_mat(A);
    
    cout << "LU decompose" << endl;
    LU(A, L, U);

    //Print_mat(A);
    //Print_mat(A_original);

    //cout << "Verify" << endl;
    //verify(A_original, L, U, N);

    Free_mat(A, N);
    Free_mat(L, N);
    Free_mat(U, N);
    return 0;
}
