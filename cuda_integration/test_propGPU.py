import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np; import tensorflow as tf


def multi_area_propagation_gpu(input_domain, net_model, thread_number=8):

	"""
	Propagation of the input domain through the network to obtain the OVERESTIMATION of the output bound. 
	The process is performed applying the linear combination node-wise and the necessary activation functions.
	The process is on GPU, completely parallelized on NVIDIA CUDA GPUs and c++ code. 

	Parameters
	----------
		input_domain : list 
			the input domain expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
			(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
		net_model : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
		thread_number : int
			number of CUDA thread to use for each CUDA block, the choice is free and does not effect the results, 
			can however effect the performance

	Returns:
	--------
		reshaped_bound : list
			the propagated bound in the same format of the input domain (3-dim)
	"""

	# Ignore the standard warning from CuPy
	import warnings
	warnings.filterwarnings("ignore")

	# Import the necessary library for the parallelization (Cupy) and also the c++ CUDA code.
	import cupy as cp
	from cuda_code_linear_relaxation import cuda_code

	# Load network shape, activations and weights
	layer_sizes = []
	activations = []
	full_weights = np.array([])
	full_biases = np.array([])

	# Iterate on each layer of the network, exluding the input (tf2 stuff)
	for layer in net_model.layers[1:]:

		# Obtain the activation function list
		if layer.activation == tf.keras.activations.linear: activations.append(0)
		elif layer.activation == tf.keras.activations.relu: activations.append(1)
		elif layer.activation == tf.keras.activations.tanh: activations.append(2)
		elif layer.activation == tf.keras.activations.sigmoid: activations.append(3)

		# Obtain the netowrk shape as a list
		layer_sizes.append(layer.input_shape[1])

		# Obtain all the weights for paramters and biases
		weight, bias = layer.get_weights()
		full_weights = np.concatenate((full_weights, weight.T.reshape(-1)))
		full_biases = np.concatenate((full_biases, bias.reshape(-1)))

	# Fixe last layer size
	layer_sizes.append( net_model.output.shape[1] )

	# Initialize the kernel loading the CUDA code
	my_kernel = cp.RawKernel(cuda_code, 'my_kernel')

	# Convert all the data in cupy array beore the kernel call
	max_layer_size = max(layer_sizes)
	results_cuda = cp.zeros(layer_sizes[-1] * 2 * len(input_domain), dtype=cp.float32)
	layer_sizes = cp.array(layer_sizes, dtype=cp.int32)
	activations = cp.array(activations, dtype=cp.int32)
	input_domain = cp.array(input_domain, dtype=cp.float32)
	full_weights = cp.array(full_weights, dtype=cp.float32)
	full_biases = cp.array(full_biases, dtype=cp.float32)


	# Define the number of CUDA block
	block_number = int(len(input_domain) / thread_number) + 1
	
	# Create and launch the kernel, wait for the sync of all threads
	kernel_input = (input_domain, len(input_domain), layer_sizes, len(layer_sizes), full_weights, full_biases, results_cuda, max_layer_size, activations)
	
	my_kernel((block_number, ), (thread_number, ), kernel_input)
	cp.cuda.Stream.null.synchronize()

	# Reshape the results and convert in numpy array
	reshaped_bound = cp.asnumpy(results_cuda).reshape((len(input_domain), net_model.layers[-1].output_shape[1], 2))

	#
	return reshaped_bound




input_domain = [[3.001, 5.], [4., 6.]]
net_model = tf.keras.models.load_model( "model_neg.h5", compile=False )
print(multi_area_propagation_gpu(input_domain, net_model))