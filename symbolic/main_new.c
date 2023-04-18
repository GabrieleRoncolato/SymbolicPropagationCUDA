#include <stdio.h>
#include "nnet.h"



int main( int argc, char *argv[])
{


    //char *FULL_NET_PATH = "/home/server/Desktop/test/nnet/ACASXU_run2a_1_1_batch_2000.nnet";
    char *FULL_NET_PATH = "/mnt/c/Users/ronco/Desktop/thesis_material/symbolic/my_nnet/model_new.nnet";

    printf("\nLoading model: %s\n", FULL_NET_PATH);

    int target = 0;
    struct NNet* nnet = load_network(FULL_NET_PATH, target);


    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    printf("Num layer: %d\n",numLayers);
    printf("Input size: %d\n",inputSize);
    printf("Output size: %d\n\n",outputSize);


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


    // SYMBOLIC INTERVAL PROPAGATION
    int i,j,k,layer;
    int maxLayerSize = nnet->maxLayerSize;

    float ****matrix = nnet->matrix;

    // equation is the temp equation for each layer
    float equation_upper[maxLayerSize][inputSize+1];
    float equation_lower[maxLayerSize][inputSize+1];
    float new_equation_upper[maxLayerSize][inputSize+1];
    float new_equation_lower[maxLayerSize][inputSize+1];

    memset(equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);

    float tempVal_upper, tempVal_lower;

    // create an identity matrix
    for (i=0;i<nnet->inputSize;i++) {
        equation_lower[i][i] = 1;
        equation_upper[i][i] = 1;
    }

    for (layer=0;layer<(numLayers);layer++) {

        //for (i=0;i<maxLayerSize;i++) {
        memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        //}

        for (i=0;i<nnet->layerSizes[layer+1];i++) {

            float **weights = matrix[layer][0];
            float **biases  = matrix[layer][1];

            tempVal_upper = tempVal_lower = 0.0;

            // update new_equation_upper and new_lower_equation using the equation_upper and equation_lower (at the beginning they are the identity matrices) previously updated.
            // example with small nnet, at the fist layer we have [x+y, x+y] for the first node and [3x+2y,3x+2y] for the second one. Hence, the matrix new_equation_lower = [[1,1],
            //                                                                                                                                                                [3,2]]
            // and the same new_equation_upper = [[1,1],
            //                                    [3,2]]
            // where for the equation_lower the first [1,1] are precisely the coefficient of the lower equation 1*x + 1*y ([1,1]),  while [3,2] are the coefficients
            // of the lower equation 3*x + 2*y, of the second node.
            for (j=0;j<nnet->layerSizes[layer];j++) {

                for (k=0;k<inputSize+1;k++) {
                    if (weights[i][j] >= 0) {
                        new_equation_upper[i][k] += equation_upper[j][k]*weights[i][j];
                        new_equation_lower[i][k] += equation_lower[j][k]*weights[i][j];
                    }
                    else {
                        new_equation_upper[i][k] += equation_lower[j][k]*weights[i][j];
                        new_equation_lower[i][k] += equation_upper[j][k]*weights[i][j];
                    }

                }

            }


            // here we update the temporal value for the lower and upper that will be returned as final output
            for (k=0;k<inputSize;k++) {

                if (new_equation_lower[i][k] >= 0) {
                    tempVal_lower += new_equation_lower[i][k] * input_interval.lower_matrix.data[k];
                }
                else {
                    tempVal_lower += new_equation_lower[i][k] * input_interval.upper_matrix.data[k];
                }

                if (new_equation_upper[i][k] >= 0) {
                    tempVal_upper += new_equation_upper[i][k] * input_interval.upper_matrix.data[k];
                }
                else {
                    tempVal_upper += new_equation_upper[i][k] * input_interval.lower_matrix.data[k];
                }

            }

            new_equation_lower[i][inputSize] += biases[i][0];
            new_equation_upper[i][inputSize] += biases[i][0];

            tempVal_lower += new_equation_lower[i][inputSize];
            tempVal_upper += new_equation_upper[i][inputSize];

            /* Perform ReLU */
            if (layer < (numLayers - 1)) {

                if (tempVal_lower < 0.0) {
                    tempVal_lower = 0.0;

                    memset(new_equation_upper[i], 0,sizeof(float)*(inputSize+1));
                    memset(new_equation_lower[i], 0,sizeof(float)*(inputSize+1));

                    new_equation_upper[i][inputSize] = tempVal_upper;
                }

                if (tempVal_upper < 0.0){
                    tempVal_upper = 0.0;

                    memset(new_equation_upper[i], 0,sizeof(float)*(inputSize+1));
                    memset(new_equation_lower[i], 0, sizeof(float)*(inputSize+1));
                }
            }
            else {
                output_interval.upper_matrix.data[i] = tempVal_upper;
                output_interval.lower_matrix.data[i] = tempVal_lower;
            }
        }

        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
    }

    printf("Symbolic interval propagation:\n");
    for (i=0;i<outputSize;i++) {
        printf("[ %f ", output_interval.lower_matrix.data[i]);
        printf("%f ]\n", output_interval.upper_matrix.data[i]);
        printf("Interval size: %f \n\n", output_interval.upper_matrix.data[i] - output_interval.lower_matrix.data[i]);
    }


    // free the memory
    destroy_network(nnet);

}