cuda_code = '''

extern "C" __global__ void my_kernel_relaxation(float* input_domain, int input_domain_n, int input_size, int output_size, int* layer_sizes, int layer_number, float* full_weights, 
			float* full_biases, float* results_cuda, int max_layer_size, int* activations) {

    // Copy global input_domain into local 'input_interval' array

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= input_domain_n) return;
    int area_start = thread_id * input_size * 2;

    
    float* input_interval = new float[2 * input_size]();

    for(int i = 0; i < 2 * input_size; i++){
        input_interval[i] = input_domain[area_start + i];
    }

    float* output_interval = new float[2 * output_size]();

    //Initialize equation arrays

    float* equation = new float[max_layer_size * ((input_size * 2) + 2)]();
    float* new_equation = new float[max_layer_size * ((input_size * 2) + 2)]();

    int actual_input_size = (2 * input_size) + 2;

    for(int i = 0; i < max_layer_size; i++){
        for(int j = 0; j < actual_input_size; j++){
            equation[i * actual_input_size + j] = 0;
        }
    }

    float tempVal_upper, tempVal_lower, tempVal_upper_min, tempVal_lower_max;
    float decrement_upper = 0.0;

    for (int i = 0; i < input_size; i ++) {
        equation[i * actual_input_size + i * 2] = 1;
        equation[i * actual_input_size + (i * 2) + 1] = 1;
    }

    int bias_index = 0;
    int weights_index = 0;

    int no_overestimation = 1;

    //Begin symbolic propagation
    for (int layer = 0; layer < layer_number - 1; layer++) {
        
        for(int i = 0; i < max_layer_size; i++){
            for(int j = 0; j < (input_size * 2) + 2; j += 2){
                new_equation[i * actual_input_size + j] = 0;
                new_equation[i * actual_input_size + j + 1] = 0;
            }
        }

        for (int i = 0; i < layer_sizes[layer + 1]; i++) {

            tempVal_upper = tempVal_lower = tempVal_upper_min = tempVal_lower_max = 0.0;

            for (int j = 0; j < layer_sizes[layer]; j++) {
                for (int k = 0; k < actual_input_size; k += 2) {
                    
                    if (full_weights[weights_index] >= 0) {
                        new_equation[i * actual_input_size + k + 1] += equation[j * actual_input_size + k + 1] * full_weights[weights_index];
                        new_equation[i * actual_input_size + k] += equation[j * actual_input_size + k] * full_weights[weights_index];
                    }
                    else {
                        new_equation[i * actual_input_size + k + 1] += equation[j * actual_input_size + k] * full_weights[weights_index];
                        new_equation[i * actual_input_size + k] += equation[j * actual_input_size + k + 1] * full_weights[weights_index];
                    }
                }

                weights_index += 1;
            }

            for (int k = 0; k < input_size * 2; k += 2) {
                if (new_equation[i * actual_input_size + k] >= 0) {
                    tempVal_lower += new_equation[i * actual_input_size + k] * input_interval[k];
                    tempVal_lower_max += new_equation[i * actual_input_size + k] * input_interval[k + 1];
                }
                else {
                    tempVal_lower += new_equation[i * actual_input_size + k] * input_interval[k + 1];
                    tempVal_lower_max += new_equation[i * actual_input_size + k] * input_interval[k];
                }

                if (new_equation[i * actual_input_size + k + 1] >= 0) {
                    tempVal_upper += new_equation[i * actual_input_size + k + 1] * input_interval[k + 1];
                    tempVal_upper_min += new_equation[i * actual_input_size + k + 1] * input_interval[k];
                }
                else {
                    tempVal_upper += new_equation[i * actual_input_size + k + 1] * input_interval[k];
                    tempVal_upper_min += new_equation[i * actual_input_size + k + 1] * input_interval[k + 1];
                }
            }

            new_equation[i * actual_input_size + input_size * 2] += full_biases[bias_index];
            new_equation[i * actual_input_size + (input_size * 2) + 1] += full_biases[bias_index];
            
            bias_index += 1;

            tempVal_lower += new_equation[i * actual_input_size + input_size * 2];
            tempVal_lower_max += new_equation[i * actual_input_size + input_size * 2];

            tempVal_upper += new_equation[i * actual_input_size + (input_size * 2) + 1];

            tempVal_upper_min += new_equation[i * actual_input_size + (input_size * 2) + 1];

            //linear relaxation of RELU

            if (layer < (layer_number - 2)) {
                //all information is lost, concretize both equations to 0
                if (tempVal_upper <= 0.0) {
                
                    tempVal_upper = 0.0;
                    tempVal_lower = 0.0;

                    for(int k = 0; k < actual_input_size; k += 2){
                        new_equation[i * actual_input_size + k] = 0;
                        new_equation[i * actual_input_size + k + 1] = 0;
                    }
                }
                // layers subsequent to the first instance of an overestimated node, information is partially lost, apply linear relaxation for subsequent layer case (lower equation and upper equation are potentially different)
                /*else if(no_overestimation == 0 && tempVal_lower <= 0){
                
                    float relaxation_lower = tempVal_upper / (tempVal_upper - tempVal_lower_max);
                    float relaxation_upper = tempVal_upper_min / (tempVal_upper_min - tempVal_lower);

                    //concretize lower to 0, relax upper
                    if(tempVal_lower <= 0 && tempVal_lower_max <= 0 && tempVal_upper_min <= 0 && tempVal_upper > 0){
                    
                        for(int k = 0; k < (input_size * 2); k += 2){
                            new_equation[i * actual_input_size + k] = 0;
                            new_equation[i * actual_input_size + k + 1] *= relaxation_upper;
                        }

                        new_equation[i * actual_input_size + (input_size * 2)] = 0;
                        new_equation[i * actual_input_size + (input_size * 2) + 1] -= (tempVal_lower_max * relaxation_upper);
                    }
                    //concretize lower to 0, maintain upper
                    else if(tempVal_lower <= 0 && tempVal_lower_max <= 0 && tempVal_upper_min > 0 && tempVal_upper > 0){
                    
                        for(int k = 0; k < (input_size * 2); k += 2){
                            new_equation[i * actual_input_size + k] = 0;
                        }
                                                
                        new_equation[i * actual_input_size + (input_size * 2)] = 0;
                    }
                    //relax lower, relax upper
                    else if(tempVal_lower <= 0 && tempVal_lower_max > 0 && tempVal_upper_min <= 0 && tempVal_upper > 0){
                    
                        for(int k = 0; k < (input_size * 2); k += 2){
                            new_equation[i * actual_input_size + k] *= relaxation_lower;
                            new_equation[i * actual_input_size + k + 1] *= relaxation_upper;
                        }
                    
                        new_equation[i * actual_input_size + (input_size * 2) + 1] -= (tempVal_lower_max * relaxation_upper);
                    }
                    //relax lower, maintain upper
                    else if(tempVal_lower <= 0 && tempVal_lower_max <= 0 && tempVal_upper_min > 0 && tempVal_upper > 0){
                        for(int k = 0; k < (input_size * 2); k += 2){
                            new_equation[i * actual_input_size + k] *= relaxation_lower;
                        }
                    }
                    
                }*/
                // first layer containing overestimated nodes, information is partially lost, apply linear relaxation for the first layer case (lower equation and upper equation are equal)
                else if(tempVal_lower < 0.0){ 
                
                    float relaxation = tempVal_upper / (tempVal_upper - tempVal_lower);

                    for(int k = 0; k < (input_size * 2); k += 2){
                        new_equation[i * actual_input_size + k] *= relaxation;
                        new_equation[i * actual_input_size + k + 1] *= relaxation;
                    }

                    new_equation[i * actual_input_size + (input_size * 2) + 1] -= (tempVal_lower * relaxation);

                    no_overestimation = 0;
                }
            }
            else {
                output_interval[(i * 2) + 1] = tempVal_upper - decrement_upper;
                output_interval[i * 2] = tempVal_lower;
            }
        }

        for(int i = 0; i < max_layer_size; i++){
            for(int j = 0; j < actual_input_size; j += 2){
                equation[i * actual_input_size + j] = new_equation[i * actual_input_size + j];
                equation[i * actual_input_size + j + 1] = new_equation[i * actual_input_size + j + 1];
            }
        }
    }

    // Copy local output_interval into global 'results_cuda' array
    int results_start = thread_id * output_size * 2;

    for (int i = 0; i < output_size * 2; i++){
        results_cuda[results_start + i] = output_interval[i];
    }

    //Deallocate memory
    delete[] input_interval;
    delete[] output_interval;
    delete[] equation;
    delete[] new_equation;
}

'''