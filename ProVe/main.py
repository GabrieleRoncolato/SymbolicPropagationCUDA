import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from netver.utils.colors import bcolors
from netver.verifier import NetVer


if __name__ == "__main__":
    
    # load your model
    model_str = "models_h5/model_2_56.h5"
    model = tf.keras.models.load_model( model_str, compile=False )

    # define your safety property here
    property = {
        "type" : "positive",
        "P" : [[0., 1.], [0., 1.]],
    }

    # ProVe hyperparameters
    method = "ProVe"            # select the method: for the formal analysis select "ProVe" otherwise "estimated"
    discretization = 3          # how many digitals consider for the computation
    CPU = False                 # select the hardware for the formal analysis
    propagation = 'relaxation'       # whether to use the naive or the symbolic propagation (Wang et al. 2018) for the interval propagation
    verbose = 0                 # whether to print info abount the computation of ProVe
    cloud = 1000000             # number of random state to sample for the approximate verification i.e., the "estimated" analysis


    if method == 'ProVe':
        netver = NetVer( method, model, property, rounding=discretization, cpu_only=CPU, interval_propagation=propagation, time_out_checked=0.0, reversed = False )
    else:
        netver = NetVer( "estimated", model, property, cloud_size=cloud )

    print(bcolors.OKCYAN + '\n\t#################################################################################' + bcolors.ENDC)
    print(bcolors.OKCYAN + '\t\t\t\t\tProVe hyperparameters:\t\t\t\t' + bcolors.ENDC)
    if method == 'ProVe':  
        print(bcolors.OKCYAN + f'\t method=formal, GPU={not CPU}, interval_propagation={propagation}, rounding={discretization}, verbose={verbose}\t' + bcolors.ENDC)
    else:
        print(bcolors.OKCYAN + f'\t\t\t\tmethod=estimated, cloud_size={cloud}' + bcolors.ENDC)
    print(bcolors.OKCYAN + '\t#################################################################################'+ bcolors.ENDC)


    start = time.time()
    sat, info = netver.run_verifier( verbose )
    end = time.time()

   
    if sat:
        print( bcolors.OKGREEN + "\nThe property is SAT!")
        print( f"\tTime execution: {round((end-start)/60,5)} min\n"+ bcolors.ENDC )
    else:
        print( bcolors.FAIL + "\nThe property is UNSAT!"+ bcolors.ENDC )
        print( "\t"+bcolors.WARNING+f"Input size: {model.layers[0].input_shape[0][1]}" )
        print( "\t"+bcolors.WARNING+f"Violation rate: {round(info['violation_rate'], 3)}%" )
        print( f"\tTime execution: {round((end-start)/60,5)} min\n"+ bcolors.ENDC )
