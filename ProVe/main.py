import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from netver.utils.colors import bcolors
from netver.verifier import NetVer

model_name = "model_5_09.h5"
model_path = "./acals_scalability/acas_model"

# safety property

property = {
    "type" : "decision",
    "name": "property2",
    "P" :  [[ 0.600000,  0.679858],
			[-0.500000,  0.500000],
			[-0.500000,  0.500000],
			[ 0.450000,  0.500000],
			[-0.500000, -0.450000]],
    "A": 0
}

'''
property = {
    "type" : "positive",
    "name": "property",
    "P" :  [[ 0.0,  1.0],
			[0.0,  1.0],
			[0.0,  1.0],
            [0.0,  1.0],
            [0.0,  1.0]]
}
'''

def run_prover(propagation, model_str, property):

    # load your model
    str_model_tf = f"./acas_scalability/acas_model/ACASXU_run2a_2_7_batch_2000.h5"
    #str_model_tf = "./models_scalability/model_5_09.h5"
    model = tf.keras.models.load_model( str_model_tf, compile=False )

    # ProVe hyperparameters
    method = "ProVe"            # select the method: for the formal analysis select "ProVe" otherwise "estimated"
    discretization = 3          # how many digitals consider for the computation
    CPU = False                 # select the hardware for the formal analysis
    verbose = 1                 # whether to print info abount the computation of ProVe
    cloud = 1000000             # number of random state to sample for the approximate verification i.e., the "estimated" analysis

    if method == 'ProVe':
        netver = NetVer( method, model, property, rounding=discretization, cpu_only=CPU, interval_propagation=propagation, time_out_checked=0.05, reversed = False )
    else:
        netver = NetVer( "estimated", model, property, cloud_size=cloud)

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

    time_execution = round((end-start)/60,5)
    violation_rate = round(info['violation_rate'], 3)
    input_size = model.layers[0].input_shape[0][1]

    if sat:
        print( bcolors.OKGREEN + "\nThe property is SAT!")
        print( f"\tTime execution: {time_execution} min\n"+ bcolors.ENDC )
    else:
        print( bcolors.FAIL + "\nThe property is UNSAT!"+ bcolors.ENDC )
        print( "\t"+bcolors.WARNING+f"Input size: {input_size}" )
        print( "\t"+bcolors.WARNING+f"Violation rate: {violation_rate}%" )
        print( f"\tTime execution: {time_execution} min\n"+ bcolors.ENDC )

    return sat, model.layers[0].input_shape[0][1], round(info['violation_rate'], 3)

if __name__ == "__main__":
    #run_prover("naive", f"{model_path}/{model_name}", property)
    run_prover("symbolic", f"{model_path}/{model_name}", property)
    run_prover("relaxation", f"{model_path}/{model_name}", property)