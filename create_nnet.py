import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np



def write_NNet(weights,biases, fileName):
    """ The first line is how many layers are in the network (n° hidden + the input). The next line tells the input size of each layer, as well as the output size of the network (1). The following rows with zeros do nothing usually and are skipped in parsing. After that are listed the weights and biases of each layer. """
    
    #Open the file we wish to write
    with open(fileName,'w') as f:

        f.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        #Extract the necessary information and write the header information
        numLayers = len(weights)
        inputSize = weights[0].shape[1]
        outputSize = len(biases[-1])
        maxLayerSize = inputSize
        
        # Find maximum size of any hidden layer
        for b in biases:
            if len(b)>maxLayerSize :
                maxLayerSize = len(b)

        # Write data to header 
        f.write("%d,%d,%d,%d,\n" % (numLayers,inputSize,outputSize,maxLayerSize) )
        f.write("%d," % inputSize )
        for b in biases:
            f.write("%d," % len(b) )
        f.write("\n")

        #f.write("0.0,\n")

        for _ in range(numLayers):
            f.write("0.0,")
        f.write("\n")

        ##################
        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer.
        ##################
        for w,b in zip(weights,biases):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    f.write("%.5e," % w[i][j]) #Five digits written. More can be used, but that requires more more space.
                f.write("\n")
                
            for i in range(len(b)):
                f.write("%.5e,\n" % b[i]) #Five digits written. More can be used, but that requires more more space.


def MLP(input_size, n_hiddens, size_hiddens, output_size):
    
    state_input = Input(shape=input_size, name='input')
    h = state_input

    for i in range(n_hiddens):
        h = Dense(size_hiddens, 'relu', name='hidden_' + str(i))(h)

    y = Dense(output_size, activation='linear', name='output')(h)
    model = Model(inputs=state_input, outputs=y)

    return model



if __name__ == "__main__":

    # hyperparameters
    input_size = 2
    n_hiddens = 1
    size_hiddens = 2
    output_size = 1
    manual_weights = True

    # create keras model
    model = MLP(input_size, n_hiddens, size_hiddens, output_size)

    # check the model
    model.summary()
    
    if manual_weights:

        #############################################
        ## SET MANUAL WEIGHTS FOR SPECIFIC EXAMPLE ##
        #############################################

        # change weights if you want to test a specific example
        weights = np.array([[1, 3], [1, 2]])
        bias = np.array([0, 0])
        model.layers[1].set_weights([weights,bias])

        # weights = np.array([[7, 4], [9, -5]])
        # bias = np.array([0, 0])
        # model.layers[2].set_weights([weights,bias])

        weights = np.array([[-1], [1]])
        bias = np.array([0])
        model.layers[2].set_weights([weights,bias])

    model.save("model.h5")

    # Test model with forward propagation
    print(model(np.array([[2, 5]])))


    ##################################
    ## CONVERSION FROM .h5 TO .nnet ##
    ##################################


    # Get a list of the model weights
    model_params = model.get_weights()

    # Split the network parameters into weights and biases, assuming they alternate
    weights = model_params[0::2]
    biases  = model_params[1::2]

    # Transpose weight matrices
    weights = [w.T for w in weights]
        
    # name file to convert to .nnet file
    nnetFile = 'model_new.nnet'

    # Convert the file
    write_NNet(weights,biases,nnetFile)