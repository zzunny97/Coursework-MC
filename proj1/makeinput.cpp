#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <fstream>

#define ITER 100000000
using namespace std;

int main()
{
    string filePath = "input_big.txt";
    ofstream os(filePath.data());

    for(int i=1; i<=ITER; i++) {
        int ran = rand() % 3;
        int ranran = rand() % 1000000;
        if(ran==0) {
            os << "i " << ranran << endl;
        }
        else if(ran==1) {
            int ran2 = rand() % 100;
            os << "w " << ran2 << endl;
        }
        else {
            os << "q " << ranran << endl; 
        
        }
    }
    os.close();

    return 0;
}
