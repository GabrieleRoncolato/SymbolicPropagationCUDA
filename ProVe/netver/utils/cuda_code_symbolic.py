cuda_code = '''

extern "C" __global__ void my_kernel_symbolic(float* input_domain, int input_domain_n, int* layer_sizes, int layer_number, float* full_weights, 
			float* full_biases, float* results_cuda, int max_layer_size, int* activations, float* gradients) {
            
    int input_size = layer_sizes[0];
    int output_size = layer_sizes[layer_number - 1];

    // Copy global input_domain into local 'input_interval' array

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= input_domain_n) return;
    int area_start = thread_id * input_size * 2;
    
    //allocate shared memory
    __shared__ float* shared_allocation[6144];

    int actual_input_size = (2 * input_size) + 2;

    int node_number = 0;
    for(int i = 0; i < layer_number; i++){
        node_number += layer_sizes[i];
    }

    int thread_idx = (max_layer_size * 2 + node_number + input_size * 2 + output_size * 2 + actual_input_size * max_layer_size * 2) * threadIdx.x;
    float* thread_allocation = (float*)&shared_allocation[thread_idx];

    //Initialize equation arrays

    float* input_interval = (float*)thread_allocation;
    float* output_interval = (float*)&thread_allocation[input_size * 2];
    float* equation = (float*)&thread_allocation[input_size * 2 + output_size * 2];
    float* new_equation = (float*)&thread_allocation[max_layer_size * actual_input_size + input_size * 2 + output_size * 2];

    float* concretizations = (float*)&thread_allocation[2 * max_layer_size * actual_input_size + input_size * 2 + output_size * 2];
    float* grads_temp = (float*)&thread_allocation[node_number + 2 * max_layer_size * actual_input_size + input_size * 2 + output_size * 2];

    for(int i = 0; i < 2 * input_size; i++){
        input_interval[i] = input_domain[area_start + i];
    }

    gradients = (float*)&gradients[thread_id * max_layer_size * 2];
    
    for(int i = 0; i < max_layer_size; i++){
        for(int j = 0; j < actual_input_size; j++){
            equation[i * actual_input_size + j] = 0;
        }
    }

    float tempVal_upper, tempVal_lower, tempVal_upper_min;

    for (int i = 0; i < input_size; i++) {
        equation[i * actual_input_size + i * 2] = 1;
        equation[i * actual_input_size + (i * 2) + 1] = 1;
    }

    int bias_index = 0;
    int weights_index = 0;
    int concretizations_index = 0;

    //Begin symbolic propagation
    for (int layer = 0; layer < layer_number - 1; layer++) {
        
        for(int i = 0; i < max_layer_size; i++){
            for(int j = 0; j < actual_input_size; j += 2){
                new_equation[i * actual_input_size + j] = 0;
                new_equation[i * actual_input_size + j + 1] = 0;
            }
        }

        for (int i = 0; i < layer_sizes[layer + 1]; i++) {

            tempVal_upper = tempVal_lower = tempVal_upper_min = 0.0;

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
                }
                else {
                    tempVal_lower += new_equation[i * actual_input_size + k] * input_interval[k + 1];
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
            tempVal_upper += new_equation[i * actual_input_size + (input_size * 2) + 1];
            tempVal_upper_min += new_equation[i * actual_input_size + (input_size * 2) + 1];

            //concretization of RELU

            if (layer < (layer_number - 2)) {
                if(activations[layer] == 1){
                    if (tempVal_upper <= 0.0){
                        tempVal_upper = 0.0;
                        tempVal_lower = 0.0;
                        tempVal_upper_min = 0.0;

                        for(int k = 0; k < input_size * 2; k += 2){
                            new_equation[i * actual_input_size + k] = 0;
                            new_equation[i * actual_input_size + k + 1] = 0;
                        }

                        new_equation[i * actual_input_size + (input_size * 2)] = 0;
                        new_equation[i * actual_input_size + (input_size * 2) + 1] = 0;

                        //printf("0 ");
                        concretizations[concretizations_index] = 0;
                    }
                    else if (tempVal_lower < 0.0) {
                        tempVal_lower = 0.0;

                        for(int k = 0; k < input_size * 2; k += 2){
                            new_equation[i * actual_input_size + k] = 0;
                        }

                        new_equation[i * actual_input_size + (input_size * 2)] = 0;

                        if(tempVal_upper_min <= 0){
                            tempVal_upper_min = tempVal_upper;

                            for(int k = 0; k < input_size * 2; k += 2){
                                new_equation[i * actual_input_size + k + 1] = 0;
                            }

                            new_equation[i * actual_input_size + (input_size * 2) + 1] = tempVal_upper;
                        }

                        //printf("1 ");
                        concretizations[concretizations_index] = 1;
                    }
                    else{
                        //printf("2 ");
                        concretizations[concretizations_index] = 2;
                    }
                }
                else{
                    concretizations[concretizations_index] = 2;
                }

                concretizations_index++;
            }
            else {
                output_interval[i * 2] = tempVal_lower;
                output_interval[(i * 2) + 1] = tempVal_upper;
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

    weights_index--;
    concretizations_index--;

    /*
    for(int j = output_size - 1; j >= 0; j--){
        gradients[j * 2] += output_interval[j * 2];
        gradients[j * 2 + 1] += output_interval[j * 2 + 1];
    }
    */

    for(int j = 0; j < max_layer_size; j++){
        grads_temp[j * 2] = 0;
        grads_temp[j * 2 + 1] = 0;
    }
    
    for(int j = layer_sizes[layer_number - 1] - 1; j >= 0; j--){
        for(int i = layer_sizes[layer_number - 2] - 1; i >= 0; i--){
            grads_temp[i * 2 + 1] += full_weights[weights_index];
            grads_temp[i * 2] += full_weights[weights_index];

            weights_index--;
        }
    }

    for(int j = 0; j < max_layer_size; j++){
        gradients[j * 2] = grads_temp[j * 2];
        gradients[j * 2 + 1] = grads_temp[j * 2 + 1];
    }

    for(int layer = layer_number - 3; layer >= 0; layer--){
        for(int j = 0; j < max_layer_size; j++){
            grads_temp[j * 2] = 0;
            grads_temp[j * 2 + 1] = 0;
        }

        for(int j = max_layer_size - 1; j >= layer_sizes[layer + 1]; j--){
            gradients[j * 2] = 0;
            gradients[j * 2 + 1] = 0;
        }

        for(int j = layer_sizes[layer + 1] - 1; j >= 0; j--){
            
            if(concretizations[concretizations_index] == 0){
                gradients[j * 2] = 0;
                gradients[j * 2 + 1] = 0;
            }
            else if(concretizations[concretizations_index] == 1){
                gradients[j * 2] = (gradients[j * 2] < 0) ? gradients[j * 2] : 0;
                gradients[j * 2 + 1] = (gradients[j * 2 + 1] > 0) ? gradients[j * 2 + 1] : 0;
            }

            concretizations_index--;

            for(int i = layer_sizes[layer] - 1; i >= 0; i--){
                if(full_weights[weights_index] >= 0){
                    grads_temp[i * 2 + 1] += full_weights[weights_index] * gradients[j * 2 + 1];
                    grads_temp[i * 2] += full_weights[weights_index] * gradients[j * 2];
                }
                else{
                    grads_temp[i * 2 + 1] += full_weights[weights_index] * gradients[j * 2];
                    grads_temp[i * 2] += full_weights[weights_index] * gradients[j * 2 + 1];
                }

                weights_index--;
            }
        }

        if(layer != 0){
            for(int j = 0; j < max_layer_size; j++){
                gradients[j * 2] = grads_temp[j * 2];
                gradients[j * 2 + 1] = grads_temp[j * 2 + 1];
            }
        }
        else{
            for(int j = 0; j < input_size; j++){
                gradients[j * 2] = grads_temp[j * 2];
                gradients[j * 2 + 1] = grads_temp[j * 2 + 1];
            }

            for(int j = input_size; j < max_layer_size; j++){
                gradients[j * 2] = 0;
                gradients[j * 2 + 1] = 0;
            }
        }
    }
}

'''
