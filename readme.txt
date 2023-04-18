Dependencies: 
    - Tensorflow
    - numpy
    - cupy
    - cmake

Project structure:
    - create_nnet.py: python script which can be used to create neural networks with different parameters (input size, output size, weights, ...) in h5 and nnet format

    - "symbolic" folder: multidimensional implementation of symbolic propagation in C. It is necessary to modify the hardcoded path in main_new.c to target a neural network in nnet format.
        In order to build and run this implementation use the following commands from within the "symbolic/build" folder:
            - cmake ..
            - make
            - ./main

    - "cuda_integration" folder: monodimensional implementation of symbolic propagation in CUDA.
        cuda_code.py contains the monodimensional interval propagation CUDA code, while cuda_code_symbolic.py contains the monodimensional symbolic propagation CUDA code.
        test_propGPU.py targets a neural network using a hardcoded path and verifies a given property using either interval propagation or symbolic propagation with cupy.