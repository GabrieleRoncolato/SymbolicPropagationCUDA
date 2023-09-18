import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from scripts.countingProve import CountingProve
from tqdm import tqdm

if __name__ == "__main__":
	
	property_2 = True

	if property_2:
		# net to test 
		prp_acas = ['2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_1', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '5_1', '5_2', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9']
		
		# ACAS XU SAFETY PROPERTY φ2 DEFINITION
		prp = {
				"type" : "decision",
				"P" : [
					[ 0.600000,  0.679858],
					[-0.500000,  0.500000],
					[-0.500000,  0.500000],
					[ 0.450000,  0.500000],
					[-0.500000, -0.450000]
				],
				"A" : 0
			}
	else:
		# net to test 
		prp_acas = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9']
		
		# ACAS SAFETY PROPERTY φ3 DEFINITION        
		prp = {
				"type" : "decision",
				"P" : [
					[-0.303531,  -0.298553],
					[-0.009549,   0.009549],
					[ 0.493380,   0.500000],
					[ 0.300000,   0.500000],
					[ 0.300000,  -0.450000]
				],
				"A" : 0
			}


	# Hyperparameters CountingProVe
	beta = 0.02
	T = 350
	S = 30
	m = 3000000
	rounding = 3
	GPU_available = False

	# Acas Xu information
	acas_model = True
	reversed = False if property_2 else True

	for net_name in prp_acas:

		# LOAD MODEL
		str_model_tf = f"acas_model/ACASXU_run2a_{net_name}_batch_2000.h5"
		model = tf.keras.models.load_model( str_model_tf, compile=False )

		
		print(f'\n\tTesting model: {str_model_tf}')
		print(f'\tCountingProVe hyperparameters: [S = {S}, β = {beta}, T = {T}, m = {m}, GPU={GPU_available}, confidence = {round((1 - 2 ** (-beta*(T)))*100, 2)}%]\n')

		# starting CountingProVe
		lower_violation_rate = 1
		lower_safe_rate = 1

		for t in tqdm(range(T)):

			countingProve = CountingProve( model, prp["P"], GPU_available, acas=acas_model, reversed=reversed)
			violation_t = countingProve.count( beta, S, m )

			countingProve = CountingProve( model, prp["P"], GPU_available, acas=acas_model, reversed=reversed )
			safe_t = countingProve.count( beta, S, m, count_violation=False )

			lower_violation_rate = min(lower_violation_rate, violation_t)
			lower_safe_rate = min(lower_safe_rate, safe_t)

		lower_bound = round(lower_violation_rate*100, 3)
		upper_bound = round(lower_safe_rate*100, 3)

		print( f"\nThe property is SAT? {lower_bound == 0}" )
		print(f"\tConfidence: {round((1 - 2 ** (-beta*(T)))*100, 2)}%")
		print( f"\tLower bound VR: {lower_bound}%" )
		print( f"\tUpper bound VR: {100 - upper_bound}%" )
		print( f"\tSize of the interval: {round((100 - upper_bound) - lower_bound, 2)}%" )
		print( f"\tViolation rate: {round(((100 - upper_bound) + lower_bound)/2,2)}%\n" )