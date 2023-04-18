extern "C" __global__ void my_kernel(float* input_domain, int input_domain_n, int* layer_sizes, int layer_number, float* full_weights, 
			float* full_biases, float* results_cuda, int max_layer_size, int* activations) {

	// Calculate all the bounds, node by node, for each layer. 'new_layer_values' is the current working layer, old layer is the prevoius (first step old layer is the input layer)
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_id >= input_domain_n) return;
	int area_start = thread_id * layer_sizes[0] * 2;
	
	
	//begin symbolic propagation
    int numLayers    = layer_number;
    int inputSize    = layer_sizes[0];
    int outputSize   = layer_sizes[layer_number - 1];

    float* input_interval = new float[2 * inputSize];

    for(int i = 0; i < 2 * inputSize; i++){
        input_interval[i] = input_domain[area_start + i];
    }

    float* output_interval = new float[2 * outputSize];

    float* equation = new float[max_layer_size][(inputSize * 2) + 2];
    float* new_equation = new float[max_layer_size][(inputSize * 2) + 2];

    for(int i = 0; i < maxLayerSize; i++){
        for(int j = 0; j < (2 * inputSize) + 2; j++){
            equation[i][j] = 0;
        }
    }

    float tempVal_upper, tempVal_lower;

    for (int i = 0; i < inputSize * 2; i ++) {
        equation[i][i * 2] = 1;
        equation[i][(i * 2) + 1] = 1;
    }

    int bias_index = 0;
    int weights_index = 0;

    for (int layer = 0; layer < numLayers; layer++) {

        for(int i = 0; i < maxLayerSize; i++){
            for(int j = 0; j < (input_size * 2) + 2; j += 2){
                new_equation[i][j] = 0;
                new_equation[i][j + 1] = 0;
            }
        }

        for (int i = 0; i < layerSizes[layer + 1]; i++) {

            tempVal_upper = tempVal_lower = 0.0;

            for (int j = 0; j < layerSizes[layer]; j++) {
                for (int k = 0; k < (inputSize * 2) + 2; k += 2) {
                    if (full_weights[weights_index] >= 0) {
                        new_equation[i][k + 1] += equation[j][k + 1] * full_weights[weights_index];
                        new_equation[i][k] += equation[j][k] * full_weights[weights_index];
                    }
                    else {
                        new_equation[i][k + 1] += equation[j][k] * full_weights[weights_index];
                        new_equation[i][k] += equation[j][k + 1] * full_weights[weights_index];
                    }
                }

                weights_index += 1;
            }

            for (int k = 0; k < inputSize * 2; k += 2) {
                if (new_equation[i][k] >= 0) {
                    tempVal_lower += new_equation[i][k] * input_interval[k];
                }
                else {
                    tempVal_lower += new_equation[i][k] * input_interval[k + 1];
                }

                if (new_equation[i][k + 1] >= 0) {
                    tempVal_upper += new_equation[i][k + 1] * input_interval[k + 1];
                }
                else {
                    tempVal_upper += new_equation[i][k + 1] * input_interval[k];
                }
            }

            new_equation[i][inputSize * 2] += full_biases[bias_index];
            new_equation[i][(inputSize * 2) + 1] += full_biases[bias_index];

            bias_index += 1;

            tempVal_lower += new_equation[i][inputSize * 2];
            tempVal_upper += new_equation[i][(inputSize * 2) + 1];

            if (layer < (numLayers - 1)) {

                if (tempVal_lower < 0.0) {
                    tempVal_lower = 0.0;

                    for(int k = 0; k < (inputSize * 2) + 2; k += 2){
                        new_equation[i][k] = 0;
                        new_equation[i][k + 1] = 0;
                    }

                    new_equation[i][(inputSize * 2) + 1] = tempVal_upper;
                }

                if (tempVal_upper < 0.0){
                    tempVal_upper = 0.0;

                    for(int k = 0; k < (inputSize * 2) + 2; k += 2){
                        new_equation[i][k] = 0;
                        new_equation[i][k + 1] = 0;
                    }
                }
            }
            else {
                output_interval[i + 1] = tempVal_upper;
                output_interval[i] = tempVal_lower;
            }
        }

        for(int i = 0; i < max_layer_size; i++){
            for(int j = 0; j < (2 * inputSize) + 2; j += 2){
                equation[i][j] = new_equation[i][j];
                equation[i][j + 1] = new_equation[i][j + 1];
            }
        }
    }

	// Step 3: copy the local output layer in the global 'results_cuda' array
	int results_start = thread_id * outputSize * 2;

	for (int i = 0; i < layer_sizes[layer_number - 1] * 2; i++){
        results_cuda[results_start + i] = output_interval[i];
    }
}
