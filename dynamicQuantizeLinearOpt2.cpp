#include <iostream>
#include <limits>
#include <cassert>
#include <algorithm>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <string>
#include <stdlib.h>

using namespace std;
#define NUM_TESTS 1000
#define NUM_THREADS 32

// Input ip_tensor: float
// Output op_tensor: uint8
template <class T>
T saturate (T ip){
    return min(max(ip, (T) 0), (T) 255);
}

void generateInput(vector<float> &x){

    random_device rd;
    mt19937 generate(rd());
    uniform_real_distribution<float> rand(0.0, 1.0);

    for (float i = 0; i < x.size(); i++) {
        x[i] = rand(generate);
    }

}

void printResult(vector<uint8_t> &y){
    for (int i=0; i<y.size(); i++) {
        cout<<unsigned(y[i])<<endl;
    }
}

void openmp_1(vector<float> &x, vector<uint8_t> &y, unsigned long long const& numElts){

    float min_x, max_x = x[0];
    #pragma omp parallel for reduction(min:min_x) reduction(max:max_x)
    for (unsigned long long i=1; i<numElts; i++) {
        min_x = min(min_x, x[i]);
        max_x = max(max_x, x[i]);
    }

    float const y_scale = (max(0.0f, max_x) - min(0.0f, min_x))/255;
    
    float intermediate_zero_point = -min_x/y_scale;

    uint8_t y_zero_point = saturate(nearbyint(intermediate_zero_point));
    // cout<<unsigned(y_zero_point)<<endl;

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (unsigned long long i=0; i<numElts; i++) {
        y[i] = saturate(nearbyint(x[i] / y_scale) + y_zero_point);
    }

}

int main(int argc, char *argv[]){
    assert(8 * sizeof (float) == 32);           // check if float is 32-bit
    assert(8 * sizeof (uint8_t) == 8);          // check if uint8_t is 8-bit

    vector<float> x;				// input tensor
    vector<uint8_t> y;                          // output tensor
    unsigned long long numElts; 

    unsigned long long test = strtoul(argv[1], 0, 0);    
    if (test < 3) {
    	if (test == 1) x = {-1, -2.1, -1.3, -2.5, -3.34, -4.0};                          
    	else if (test == 2) x = {0, 2, -3, -2.5, 1.34, 0.5};                          
    	else x = {1, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345};                          
    	numElts = x.size();
    } else {
    	numElts = test;
    	x.resize(numElts, 0);
    	generateInput(x);
    }
    cout<<"Total number of input elements: "<<numElts<<endl;

    y.resize(x.size());
    clock_t start_time = omp_get_wtime();
    double total_time = 1, crtTime = 1;

    for (int test = 0; test < NUM_TESTS; test++){
        start_time = omp_get_wtime();
    	openmp_1(x, y, numElts);
        crtTime = (omp_get_wtime() - start_time) / (double)CLOCKS_PER_SEC;
        total_time = total_time > crtTime ? crtTime : total_time;
    }

    cout<<"Optimization 2 time: "<<total_time<<endl;
    if (test < 3){
        printResult(y);
    }

    return 0;
}
