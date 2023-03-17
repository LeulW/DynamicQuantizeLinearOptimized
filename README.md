# DynamicQuantizeLinearOptimized
Software-based optimization of the DynamicQuantizeLinear operator (https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear)

  `Optimization 1`: multi-threaded implementation via OpenMP

  `Optimization 2`: multiple multi-threaded implementation via OpenMP

  `Optimization 3`: (a) thread-local storage to avoid thread synchronization on shared data (b) padding to reduce false sharing

  `Optimization 4`: replaces the per-thread interleaved accesses to sequential accesses

Use the run.sh script to test and compare the performance the above optimizations

                      `bash run.sh {num_elements}`

It compiles the *.cpp implementations and run them on a randomly filled vector that contains num_elements elements

For ex. `bash run.sh 1000000` runs and emits their performance on a vector that has 1000000 elements
