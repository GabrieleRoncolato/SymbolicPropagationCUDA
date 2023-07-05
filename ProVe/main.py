import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from netver.utils.colors import bcolors
from netver.verifier import NetVer

model_name = "model_4_80.h5"
model_path = "./models_h5"

# safety property
property1 = {
    "type" : "positive",
    "name": "property1",
    #"P" : [[0.5, 1.], [0., 0.3], [0.47, 0.83], [0.72, 0.82], [0.9, 1.]],
    #"P" : [[0.5, 1.], [0., 0.3], [0.47, 0.83], [0.72, 0.82], [0.9, 1.], [0.3, 0.4], [0.1, 0.2]],
    "P" : [[0.2, 0.4], [0.1, 0.2], [0.7, 0.8], [0.9, 1.]],
}

def run_prover(propagation, model_str, property):
    # load your model

    model = tf.keras.models.load_model( model_str, compile=False )
    

    # ProVe hyperparameters
    method = "ProVe"            # select the method: for the formal analysis select "ProVe" otherwise "estimated"
    discretization = 3          # how many digitals consider for the computation
    CPU = False                 # select the hardware for the formal analysis
    verbose = 1                 # whether to print info abount the computation of ProVe
    cloud = 1000000             # number of random state to sample for the approximate verification i.e., the "estimated" analysis

    if method == 'ProVe':
        netver = NetVer( method, model, property, rounding=discretization, cpu_only=CPU, interval_propagation=propagation, time_out_checked=0., reversed = True )
    else:
        netver = NetVer( "estimated", model, property, cloud_size=cloud )

    print(bcolors.OKCYAN + '\n\t#################################################################################' + bcolors.ENDC)
    print(bcolors.OKCYAN + '\t\t\t\t\tProVe hyperparameters:\t\t\t\t' + bcolors.ENDC)
    if method == 'ProVe':  
        print(bcolors.OKCYAN + f'\t method=formal, GPU={not CPU}, interval_propagation={propagation}, rounding={discretization}, verbose={verbose}\t' + bcolors.ENDC)
    else:
        print(bcolors.OKCYAN + f'\t\t\t\tmethod=estimated, cloud_size={cloud}' + bcolors.ENDC)
    print(bcolors.OKCYAN + '\t#################################################################################'+ bcolors.ENDC)

    log_file = None
    if verbose > 1:
        if not os.path.exists(f"./logs/{model_name}"):
            os.makedirs(f"./logs/{model_name}")
        log_file = open(f"./logs/{model_name}/{propagation}_{property['name']}.txt", "w")
        log_file.write(f"Checking input: {property['P']}\n")

    start = time.time()
    sat, info = netver.run_verifier( verbose, log_file )
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

    if verbose > 1:
        if sat:
            log_file.write("The property is SAT!\n")
            log_file.write(f"Time execution: {time} min \n")
        else:
            log_file.write("The property is UNSAT!\n")
            log_file.write(f"Input size: {input_size}\n")
            log_file.write(f"Violation rate: {violation_rate}\n")
            log_file.write(f"Time execution: {time_execution} min\n")

        log_file.close()

    return sat, model.layers[0].input_shape[0][1], round(info['violation_rate'], 3)

if __name__ == "__main__":
    #run_prover("naive", f"{model_path}/{model_name}", property1)
    run_prover("symbolic", f"{model_path}/{model_name}", property1)
    #run_prover("relaxation", f"{model_path}/{model_name}", property1)