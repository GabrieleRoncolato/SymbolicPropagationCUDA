//
// Created by server on 1/24/23.
//
#include "matrix.h"
#include <string.h>
#include "interval.h"

#ifndef TEST_NNET_H
#define TEST_NNET_H


/* which property to test */
extern int PROPERTY;

/* log file */
extern char *LOG_FILE;
extern FILE *fp;

typedef int bool;
enum { false, true };


/*
 * Network instance modified from Reluplex
 * malloc all the memory needed for network
 */
struct NNet
{
    int symmetric;
    int numLayers;
    int inputSize;
    int outputSize;
    int maxLayerSize;
    int *layerSizes;

    float *mins;
    float *maxes;
    float *means;
    float *ranges;
    float ****matrix;

    struct Matrix* weights;
    struct Matrix* bias;

    int target;
    int *feature_range;
    int feature_range_length;
    int split_feature;
};


/* load the acas network from file */
struct NNet *load_network_acas(const char* filename, int target);

/* load the network from file */
struct NNet *load_network(const char *filename, int target);

/* free all the memory for the network */
void destroy_network(struct NNet *network);


/*
 * Uses for loop to calculate the output
 * 0.00002607 sec for one run with one core
*/
int forward_propagation(struct NNet *network, struct Matrix *input, struct Matrix *output);


/*
 * Uses for loop to calculate the interval output
 * 0.000091 sec for one run with one core
*/
int naive_interval_propagation(struct NNet *network, struct Interval *input, struct Interval *output);


/*
 * Uses for loop with equation to calculate the interval output
 * 0.000229 sec for one run with one core
*/
int symbolic_interval_propagation(struct NNet *network, struct Interval *input, struct Interval *output);




#endif //TEST_NNET_H
