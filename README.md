# DynamicQuantizeLinearOptimized
Software optimization of the DynamicQuantizeLinear operator

Optimization 1: multi-threaded implementation via OpenMP
Optimization 2: multiple multi-threaded implementation via OpenMP
Optimization 3: (a) thread-local storage to avoid thread synchronization on shared data (b) padding to reduce false sharing
Optimization 4: sequential accesses per thread in place interleaved accesses

use the run.sh script to test and compare the performance the above optimization

`bash run.sh {num_elements}`

It compiles the *.cpp implementations and run them on a radomly filled vector that contains num_elements elements

For ex. to test their performance on a vector that contains 1000000 element: `bash run.sh 1000000`
