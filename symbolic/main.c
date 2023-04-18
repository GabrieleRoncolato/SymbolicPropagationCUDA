#include <stdio.h>
#include "nnet.h"



int main( int argc, char *argv[])
{


    //char *FULL_NET_PATH = "/home/server/Desktop/test/nnet/ACASXU_run2a_1_1_batch_2000.nnet";
    char *FULL_NET_PATH = "/Users/luca/Desktop/symbolic/test.nnet";

    int target = 0;
    struct NNet* nnet = load_network(FULL_NET_PATH, target);


    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    printf("num layer: %d\n",numLayers);
    printf("inputSize: %d\n",inputSize);
    printf("outputSize: %d\n\n",outputSize);



    float input_test[] = {0.1, 0.5}; //, 0.7, 1.5, 0.4};

    struct Matrix input_t = {input_test, 1, nnet->inputSize};
    float u[inputSize], l[inputSize];


    // property definition
    float upper[] = {5, 6};
    float lower[] = {1, 4};

    memcpy(u, upper, sizeof(float)*inputSize);
    memcpy(l, lower, sizeof(float)*inputSize);

    struct Matrix input_upper = {u,1,nnet->inputSize};
    struct Matrix input_lower = {l,1,nnet->inputSize};
    struct Interval input_interval = {input_lower, input_upper};

    float o[nnet->outputSize];
    struct Matrix output = {o, outputSize, 1};

    float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
    struct Interval output_interval = {(struct Matrix){o_lower, outputSize, 1},
                                        (struct Matrix){o_upper, outputSize, 1}};

    // forward propagation of input_t
    forward_propagation(nnet, &input_t, &output);
    //printMatrix(&output);

    // naive interval propagation
    naive_interval_propagation(nnet, &input_interval, &output_interval);

    printf("\n\n");

    // symbolic interval propagation
    symbolic_interval_propagation(nnet, &input_interval, &output_interval);

    // free the memory
    destroy_network(nnet);

}