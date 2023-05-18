import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from netver.utils.colors import bcolors
from netver.verifier import NetVer

model_name = "model_2_68.h5"
model_path = "models_h5"
log = True

# safety property
property = {
    "type" : "positive",
    "P" : [[1., 4.], [5., 6.]],
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

    return sat, model.layers[0].input_shape[0][1], round((end-start)/60,5), round(info['violation_rate'], 3)

if __name__ == "__main__":
    sat_naive, input_size_naive, time_execution_naive, violation_rate_naive = run_prover("naive", f"{model_path}/{model_name}", property)
    sat_relax, input_size_relax, time_execution_relax, violation_rate_relax = run_prover("relaxation", f"{model_path}/{model_name}", property)

    if log:
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        
        log_file = open(f"./logs/{model_name}.txt", "w")

        log_file.write("NAIVE\n")

        if sat_naive:
            log_file.write("The property is SAT!")
            log_file.write(f"\tTime execution: {time_execution_naive} min\n")
        else:
            log_file.write("The property is UNSAT!")
            log_file.write( "\t"+f"Input size: {input_size_naive}" )
            log_file.write( "\t"+f"Violation rate: {violation_rate_naive}%" )
            log_file.write( f"\tTime execution: {time_execution_naive} min\n" )

        log_file.write("SYMBOLIC LINEAR RELAXATION\n")

        if sat_relax:
            log_file.write("The property is SAT!")
            log_file.write(f"\tTime execution: {time_execution_relax} min\n")
        else:
            log_file.write("The property is UNSAT!")
            log_file.write( "\t"+f"Input size: {input_size_relax}" )
            log_file.write( "\t"+f"Violation rate: {violation_rate_relax}%" )
            log_file.write( f"\tTime execution: {time_execution_relax} min\n" )