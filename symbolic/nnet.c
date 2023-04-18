//
// Created by server on 1/24/23.
//
#include "nnet.h"


int PROPERTY = 5;
char *LOG_FILE = "logs/log.txt";
FILE *fp;


/*
 * Load_network is a function modified from Reluplex
 * It takes in a nnet filename with path and load the
 * network from the file
 * Outputs the NNet instance of loaded network.
 */
struct NNet *load_network_acas(const char* filename, int target)
{

    FILE *fstream = fopen(filename,"r");

    if (fstream == NULL) {
        printf("Wrong network!\n");
        exit(1);
    }

    int bufferSize = 10240;

    char *buffer = (char*)malloc(sizeof(char)*bufferSize);

    char *record, *line;
    int i=0, layer=0, row=0, j=0, param=0;

    struct NNet *nnet = (struct NNet*)malloc(sizeof(struct NNet));

    nnet->target = target;

    line=fgets(buffer,bufferSize,fstream);

    while (strstr(line, "//") != NULL) {
        line = fgets(buffer,bufferSize,fstream);
    }

    record = strtok(line,",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL,",\n"));
    nnet->outputSize = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    nnet->layerSizes = (int*)malloc(sizeof(int)*(nnet->numLayers+1));

    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");


    for (i = 0;i<((nnet->numLayers)+1);i++) {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }


    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    nnet->symmetric = atoi(record);


    nnet->mins = (float*)malloc(sizeof(float)*nnet->inputSize);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

    for (i = 0;i<(nnet->inputSize);i++) {
        nnet->mins[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }


    nnet->maxes = (float*)malloc(sizeof(float)*nnet->inputSize);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

    for (i = 0;i<(nnet->inputSize);i++) {
        nnet->maxes[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }



    nnet->means = (float*)malloc(sizeof(float)*(nnet->inputSize+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

    for (i = 0;i<((nnet->inputSize)+1);i++) {
        nnet->means[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }


    nnet->ranges = (float*)malloc(sizeof(float)*(nnet->inputSize+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

    for (i = 0;i<((nnet->inputSize)+1);i++) {
        nnet->ranges[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }


    nnet->matrix = (float****)malloc(sizeof(float *)*nnet->numLayers);

    for (layer = 0;layer<(nnet->numLayers);layer++) {
        nnet->matrix[layer] =\
                (float***)malloc(sizeof(float *)*2);
        nnet->matrix[layer][0] =\
                (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
        nnet->matrix[layer][1] =\
                (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);

        for (row = 0;row<nnet->layerSizes[layer+1];row++) {
            nnet->matrix[layer][0][row] =\
                    (float*)malloc(sizeof(float)*nnet->layerSizes[layer]);
            nnet->matrix[layer][1][row] = (float*)malloc(sizeof(float));
        }

    }


    layer = 0;
    param = 0;
    i=0;
    j=0;

    char *tmpptr=NULL;

    float w = 0.0;

    while ((line = fgets(buffer,bufferSize,fstream)) != NULL) {

        if (i >= nnet->layerSizes[layer+1]) {

            if (param==0) {
                param = 1;
            }
            else {
                param = 0;
                layer++;
            }

            i=0;
            j=0;
        }

        record = strtok_r(line,",\n", &tmpptr);

        while (record != NULL) {
            w = (float)atof(record);
            nnet->matrix[layer][param][i][j] = w;
            j++;
            record = strtok_r(NULL, ",\n", &tmpptr);
        }

        tmpptr=NULL;
        j=0;
        i++;
    }

    float orig_weights[nnet->maxLayerSize];
    float orig_bias;

    struct Matrix *weights=malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers*sizeof(struct Matrix));

    for (int layer=0;layer<nnet->numLayers;layer++) {
        weights[layer].row = nnet->layerSizes[layer];
        weights[layer].col = nnet->layerSizes[layer+1];
        weights[layer].data = (float*)malloc(sizeof(float)\
                    * weights[layer].row * weights[layer].col);

        int n=0;

        if (PROPERTY != 1) {

            /* weights in the last layer minus the weights of true label output. */
            if (layer == nnet->numLayers-1) {
                orig_bias = nnet->matrix[layer][1][nnet->target][0];
                memcpy(orig_weights, nnet->matrix[layer][0][nnet->target],\
                            sizeof(float)*nnet->maxLayerSize);

                for (int i=0;i<weights[layer].col;i++) {

                    for (int j=0;j<weights[layer].row;j++) {
                        weights[layer].data[n] =\
                                nnet->matrix[layer][0][i][j]-orig_weights[j];
                        n++;
                    }

                }

                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);

                for (int i=0;i<bias[layer].col;i++) {
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0]-orig_bias;
                }
            }
            else {

                for (int i=0;i<weights[layer].col;i++) {

                    for (int j=0;j<weights[layer].row;j++) {
                        weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                        n++;
                    }

                }

                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float) *\
                                        bias[layer].col);

                for (int i=0;i<bias[layer].col;i++) {
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0];
                }

            }
        }
        else {

            for (int i=0;i<weights[layer].col;i++) {

                for (int j=0;j<weights[layer].row;j++) {
                    weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                    n++;
                }

            }

            bias[layer].col = nnet->layerSizes[layer+1];
            bias[layer].row = (float)1;
            bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);

            for (int i=0;i<bias[layer].col;i++) {
                bias[layer].data[i] = nnet->matrix[layer][1][i][0];
            }

        }

    }

    nnet->weights = weights;
    nnet->bias = bias;

    free(buffer);
    fclose(fstream);

    return nnet;

}

/*
 * Load_network is a function modified from Load_network_acas
 * It takes in a nnet filename with path and load the
 * network from the file, without means and ranges
 * Outputs the NNet instance of loaded network.
 */
struct NNet *load_network(const char* filename, int target)
{

    FILE *fstream = fopen(filename,"r");


    if (fstream == NULL) {
        printf("Wrong network!\n");
        exit(1);
    }

    int bufferSize = 10240;

    char *buffer = (char*)malloc(sizeof(char)*bufferSize);


    char *record, *line;
    int i=0, layer=0, row=0, j=0, param=0;


    struct NNet *nnet = (struct NNet*)malloc(sizeof(struct NNet));



    nnet->target = target;

    line=fgets(buffer,bufferSize,fstream);

    while (strstr(line, "//") != NULL) {
        line = fgets(buffer,bufferSize,fstream);
    }

    record = strtok(line,",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL,",\n"));
    nnet->outputSize = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    nnet->layerSizes = (int*)malloc(sizeof(int)*(nnet->numLayers+1));



    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");


    for (i = 0;i<((nnet->numLayers)+1);i++) {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }


    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    nnet->symmetric = atoi(record);

    nnet->matrix = (float****)malloc(sizeof(float *)*nnet->numLayers);



    for (layer = 0;layer<(nnet->numLayers);layer++) {
        nnet->matrix[layer] = (float***)malloc(sizeof(float *)*2);
        nnet->matrix[layer][0] = (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
        nnet->matrix[layer][1] = (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);

        for (row = 0;row<nnet->layerSizes[layer+1];row++) {
            nnet->matrix[layer][0][row] =\
                    (float*)malloc(sizeof(float)*nnet->layerSizes[layer]);
            nnet->matrix[layer][1][row] = (float*)malloc(sizeof(float));
        }

    }

    layer = 0;
    param = 0;
    i=0;
    j=0;

    char *tmpptr=NULL;

    float w = 0.0;

    while ((line = fgets(buffer,bufferSize,fstream)) != NULL) {

        if (i >= nnet->layerSizes[layer+1]) {

            if (param==0) {
                param = 1;
            }
            else {
                param = 0;
                layer++;
            }

            i=0;
            j=0;
        }

        record = strtok_r(line,",\n", &tmpptr);

        while (record != NULL) {
            w = (float)atof(record);
            nnet->matrix[layer][param][i][j] = w;
            j++;
            record = strtok_r(NULL, ",\n", &tmpptr);
        }

        tmpptr=NULL;
        j=0;
        i++;
    }

    float orig_weights[nnet->maxLayerSize];
    float orig_bias;

    struct Matrix *weights=malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers*sizeof(struct Matrix));

    for (int layer=0;layer<nnet->numLayers;layer++) {
        weights[layer].row = nnet->layerSizes[layer];
        weights[layer].col = nnet->layerSizes[layer+1];
        weights[layer].data = (float*)malloc(sizeof(float)\
                    * weights[layer].row * weights[layer].col);

        int n=0;

        if (PROPERTY != 1) {

            /* weights in the last layer minus the weights of true label output. */
            if (layer == nnet->numLayers-1) {
                orig_bias = nnet->matrix[layer][1][nnet->target][0];
                memcpy(orig_weights, nnet->matrix[layer][0][nnet->target],\
                            sizeof(float)*nnet->maxLayerSize);

                for (int i=0;i<weights[layer].col;i++) {

                    for (int j=0;j<weights[layer].row;j++) {
                        weights[layer].data[n] =\
                                nnet->matrix[layer][0][i][j]-orig_weights[j];
                        n++;
                    }

                }

                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);

                for (int i=0;i<bias[layer].col;i++) {
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0]-orig_bias;
                }
            }
            else {

                for (int i=0;i<weights[layer].col;i++) {

                    for (int j=0;j<weights[layer].row;j++) {
                        weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                        n++;
                    }

                }

                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float) *\
                                        bias[layer].col);

                for (int i=0;i<bias[layer].col;i++) {
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0];
                }

            }
        }
        else {

            for (int i=0;i<weights[layer].col;i++) {

                for (int j=0;j<weights[layer].row;j++) {
                    weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                    n++;
                }

            }

            bias[layer].col = nnet->layerSizes[layer+1];
            bias[layer].row = (float)1;
            bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);

            for (int i=0;i<bias[layer].col;i++) {
                bias[layer].data[i] = nnet->matrix[layer][1][i][0];
            }

        }

    }

    nnet->weights = weights;
    nnet->bias = bias;

    free(buffer);
    fclose(fstream);

    return nnet;

}


/*
 * destroy_network is a function modified from Reluplex
 * It release all the memory mallocated to the network instance
 * It takes in the instance of nnet
 */
void destroy_network(struct NNet *nnet)
{

    int i=0, row=0;
    if (nnet != NULL) {

        for (i=0;i<(nnet->numLayers);i++) {

            for (row=0;row<nnet->layerSizes[i+1];row++) {
                free(nnet->matrix[i][0][row]);
                free(nnet->matrix[i][1][row]);
            }

            free(nnet->matrix[i][0]);
            free(nnet->matrix[i][1]);
            free(nnet->weights[i].data);
            free(nnet->bias[i].data);
            free(nnet->matrix[i]);
        }

        free(nnet->weights);
        free(nnet->bias);
        free(nnet->layerSizes);
        free(nnet->mins);
        free(nnet->maxes);
        free(nnet->means);
        free(nnet->ranges);
        free(nnet->matrix);
        free(nnet);
    }

}



/*
 * Concrete forward propagation with for loops
 * It takes in network and concrete input matrix.
 * Outputs the concrete outputs.
 */
int forward_propagation(struct NNet *network, struct Matrix *input, struct Matrix *output)
{

    int i,j,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    float ****matrix = nnet->matrix;

    float tempVal;
    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];

    for (i=0;i<nnet->inputSize;i++) {
        z[i] = input->data[i];
    }

    for (layer = 0;layer<numLayers;layer++) {

        for (i=0;i<nnet->layerSizes[layer+1];i++) {
            float **weights = matrix[layer][0];
            float **biases  = matrix[layer][1];
            tempVal = 0.0;

            for (j=0;j<nnet->layerSizes[layer];j++) {
                tempVal += z[j]*weights[i][j];

            }

            tempVal += biases[i][0];

            //Perform ReLU
            if (tempVal < 0.0 && layer < (numLayers - 1)) {
                // printf( "doing RELU on layer %u\n", layer );
                tempVal = 0.0;
            }

            a[i]=tempVal;
        }

        for (j=0;j<nnet->maxLayerSize;j++) {
            z[j] = a[j];
        }

    }

    for (i=0; i<outputSize; i++) {
        output->data[i] = a[i];
    }

    return 1;

}


/*
 * Naive interval propagation with for loops.
 * It takes in network and input interval.
 * Outputs the estimated output range.
 */
int naive_interval_propagation(struct NNet *network,struct Interval *input,struct Interval *output)
{

    int i,j,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    float ****matrix = nnet->matrix;

    // Printing the values
    //for (i = 0; i < numLayers; i++)
    //{
    //    for (j = 0; j < 2; j++)
    //    {
    //        for (int k = 0; k < nnet->layerSizes[i+1]; k++)
    //        {
    //
    //            for (int l = 0; l < nnet->layerSizes[i]; l++)
    //            {
    //                printf("Value of matrix[%d][%d][%d][%d] is %lf\n",i,j,k,l, nnet->matrix[i][j][k][l]);
    //            }
    //        }
    //    }
    //}



    float tempVal_upper, tempVal_lower;
    float z_upper[nnet->maxLayerSize];
    float z_lower[nnet->maxLayerSize];
    float a_upper[nnet->maxLayerSize];
    float a_lower[nnet->maxLayerSize];

    for (i=0;i < nnet->inputSize;i++) {
        z_upper[i] = input->upper_matrix.data[i];
        z_lower[i] = input->lower_matrix.data[i];
    }

    for (layer = 0;layer<(numLayers);layer++) {

        for (i=0;i<nnet->layerSizes[layer+1];i++) {
            float **weights = matrix[layer][0];
            float **biases  = matrix[layer][1];
            tempVal_upper = tempVal_lower = 0.0;

            for (j=0;j<nnet->layerSizes[layer];j++) {

                if (weights[i][j] >= 0) {
                    tempVal_upper += z_upper[j]*weights[i][j];
                    tempVal_lower += z_lower[j]*weights[i][j];
                }
                else {
                    tempVal_upper += z_lower[j]*weights[i][j];
                    tempVal_lower += z_upper[j]*weights[i][j];
                }

            }

            tempVal_lower += biases[i][0];
            tempVal_upper += biases[i][0];

            if (layer < (numLayers - 1)) {

                if (tempVal_lower < 0.0){
                    tempVal_lower = 0.0;
                }

                if (tempVal_upper < 0.0){
                    tempVal_upper = 0.0;
                }

            }

            a_upper[i] = tempVal_upper;
            a_lower[i] = tempVal_lower;

        }

        for (j=0;j<nnet->maxLayerSize;j++) {
            z_upper[j] = a_upper[j];
            z_lower[j] = a_lower[j];
        }

    }

    printf("Naive interval propagation:\n");
    for (i=0;i<outputSize;i++) {
        output->upper_matrix.data[i] = a_upper[i];
        output->lower_matrix.data[i] = a_lower[i];

        printf("[ %f ", output->lower_matrix.data[i]);
        printf("%f ]\n", output->upper_matrix.data[i]);
        printf("difference: %f \n", output->upper_matrix.data[i] - output->lower_matrix.data[i]);
    }

    return 1;
}


/*
 * Symbolic interval propagation with for loops.
 * It takes in network and input interval.
 * Outputs the estimated output range.
 */
int symbolic_interval_propagation(struct NNet *network,struct Interval *input,struct Interval *output)
{

    int i,j,k,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize   = nnet->maxLayerSize;

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
                    tempVal_lower += new_equation_lower[i][k] * input->lower_matrix.data[k];
                }
                else {
                    tempVal_lower += new_equation_lower[i][k] * input->upper_matrix.data[k];
                }

                if (new_equation_upper[i][k] >= 0) {
                    tempVal_upper += new_equation_upper[i][k] * input->upper_matrix.data[k];
                }
                else {
                    tempVal_upper += new_equation_upper[i][k] * input->lower_matrix.data[k];
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
                output->upper_matrix.data[i] = tempVal_upper;
                output->lower_matrix.data[i] = tempVal_lower;
            }

        }

        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);

    }

    printf("Symbolic interval propagation:\n");
    for (i=0;i<outputSize;i++) {
        printf("[ %f ", output->lower_matrix.data[i]);
        printf("%f ]\n", output->upper_matrix.data[i]);
        printf("difference: %f \n", output->upper_matrix.data[i] - output->lower_matrix.data[i]);
    }


    return 1;

}



