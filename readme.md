# Efficient #DNN-Verification via Parallel Symbolic Interval Propagation and Linear Relaxation

This is the Python implementation of ProVe+SLR, a tool for the #DNN-Verification Problem (i.e., the problem of computing the volume of all the areas that do (not) result in a violation for a given DNN and a safety property). ProVe+SLR is an exact approach that provides provable guarantees on the returned volume. This code is based on ProVe[A].


<div style="text-align:center;">    
<img src="images/overview.png" width="850" height="300" />
</div>


## Dependencies: 
    - Tensorflow 2.16.1
    - numpy 1.26.4
    - cupy 13.1.0
    - psutil 5.9.8
    - cmake

## Definition of the properties
Properties can be defined with 3 different formulations:

### PQ
Following the definition of Marabou [Katz et al.], given an input property P and an output property Q, the property is verified if for each *x* in P it follows that N(x) is in Q *(i.e., if the input belongs to the interval P, the output must belongs to the interval Q)*.
```python
property = {
	"type" : "PQ",
	"P" : [[0.1, 0.34531], [0.7, 1.1]],
	"Q" : [[0.0, 0.2], [0.0, 0.2]]
}
```

### Decision
Following the definition of ProVe [Corsi et al.], given an input property P and an output node A corresponding to an action, the property is verified if for each *x* in P it follows that the action A will never be selected *(i.e., if the input belongs to the interval P, the output of node A is never the one with the highest value)*.
```python
property = {
	"type" : "decision",
	"P" : [[0.1, 0.3], [0.7, 1.1]],
	"A" : 1
}
```

### Positive
Following the definition of α,β-CROWN [Wang et al.], given an input property P the output of the network is non negative *(i.e., if the input belongs to the interval P, the output of each node is greater or equals zero)*
```python
property = {
	"type" : "positive",
	"P" : [[0.1, 0.3], [0.7, 1.1]]
}
```


## Parameters
ProVe will use the default parameters for the formal analysis. You can change all the parameters when create the NetVer object as follows: 
```python
from netver.main import NetVer
netver = NetVer(algorithm_key, model, memory_limit=0, disk_limit=0, semi_formal=True, rounding=3)
```
Follow a list of the available parameters (with the default value):
```python
# Common to all the algorithms
time_out_cycle = 35 #timeout on the number of cycles
time_out_checked = 0 #timeout on the checked area. If the unproved area is less than this value, the algorithm stops returning the residual as a violation
memory_limit=0 #upper threshold for the amount of virtual memory to occupy during parallel computations, 0 indicates the current amount of free available virtual memory as the upper threshold
disk_limit=0 #upper threshold for the amount of disk space to occupy when saving partial results, 0 indicates the current amount of free disk space as the upper threshold
rounding = None #rounding value for the input domain (P)

# Only for Estimated
cloud_size = 10000 #indicates the size of the point cloud for the method
```


## Results of the analysis
The analysis returns two values, SAT and info. *SAT* is true if the property is respected, UNSAT otherwise; *value* is a dictionary that contains different values based on the used algorithm:

- counter_example: a counter-example that falsifies the property 
- violation_rate: the violation rate, i.e., the percentage of the unsafe volume of the property's domain 
- exit_reason: reason for an anticipate exit *(usually timeout)*


## Example Code
To run the example code, use *example.py* from the main folder. For the time being, we only support Keras models.
```python
import tensorflow as tf
import numpy as np
from netver.main import NetVer

def get_actor_model( ):
	inputs = tf.keras.layers.Input(shape=(2,))
	hidden_0 = tf.keras.layers.Dense(32, activation='relu', bias_initializer='random_normal')(inputs)
	hidden_1 = tf.keras.layers.Dense(32, activation='relu', bias_initializer='random_normal')(hidden_0)
	outputs = tf.keras.layers.Dense(5, activation='linear')(hidden_1)

	return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
	model = get_actor_model()

	property = {
		"type": "decision",
		"P": [[0.1, 0.3], [0.7, 1.1]],
		"A": 1
	}

	netver = NetVer("complete_prove", model, property)
	sat, info = netver.run_verifier(verbose=1)
	print( f"\nThe property is SAT? {sat}" )
	print( f"\tviolation rate: {info['violation_rate']}\n" )
```


## Contributors
*  **Gabriele Roncolato** - gabriele.roncolato@studenti.univr.it
*  **Luca Marzari** - luca.marzari@univr.it

## Reference
[A] [Formal verification of neural networks for safety-critical tasks in deep reinforcement learning](https://proceedings.mlr.press/v161/corsi21a.html) Corsi D., Marchesini E., and Farinelli A. UAI, 2021
    
If you use our verifier in your work, please kindly cite our paper:
[Scaling #DNN-Verification Tools with Efficient Bound Propagation and Parallel Computing](https://arxiv.org/pdf/2312.05890).  Marzari L., Roncolato G., and Farinelli A. AIRO, 2023
```
@incollection{marzari2023scaling,
  title={Scaling \#DNN-Verification Tools with Efficient Bound Propagation and Parallel Computing},
  author={Marzari, Luca and Roncolato, Gabriele and Farinelli, Alessandro},
  booktitle={AIRO 2023 Artificial Intelligence and Robotics 2023},
  year={2023}
}
```
