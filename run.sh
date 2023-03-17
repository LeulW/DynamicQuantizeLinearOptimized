#!/bin/bash

g++ -O3 -fopenmp -std=c++11 dynamicQuantizeLinearNaive.cpp -o dynamicQuantizeLinearNaive
g++ -O3 -fopenmp -std=c++11 dynamicQuantizeLinearOpt1.cpp -o dynamicQuantizeLinearOpt1
g++ -O3 -fopenmp -std=c++11 dynamicQuantizeLinearOpt2.cpp -o dynamicQuantizeLinearOpt2
g++ -O3 -fopenmp -std=c++11 dynamicQuantizeLinearOpt3.cpp -o dynamicQuantizeLinearOpt3
g++ -O3 -fopenmp -std=c++11 dynamicQuantizeLinearOpt4.cpp -o dynamicQuantizeLinearOpt4

./dynamicQuantizeLinearNaive $1
./dynamicQuantizeLinearOpt1 $1
./dynamicQuantizeLinearOpt2 $1
./dynamicQuantizeLinearOpt3 $1
./dynamicQuantizeLinearOpt4 $1
