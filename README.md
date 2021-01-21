# SpikeSwarmSim

## Installation
Clone this repository:
```
git clone https://github.com/r-sendra/SpikeSwarmSim.git
```
Download the simulator requirements:
```
pip3 install -r requirements.txt
```
Install python3 tkinter:
```
sudo apt-get install python3-tk
```

Additionally, if a working MPI is installed in your system:

`pip3 install mpi4py==3.0.3`


## Basic Usage
The simulator can be run from the command line using the following command:
``` 
python main.py --cfg experiment_configuration -Rv 
```
It executes the experiment defined in the JSON configuration file `experiment_configuration` (stored in `spike_swarm_sim\config`) 
in render mode (`R`) and in verbose mode (`v`). 

The following table shows the possible command line arguments with its abbreviation and description:

| Argument | Abbreviation | Description |
| :---:  | :---:  | :--- |
| `render` | `R` | Whether to run in visual or console mode. |
| `debug` | `d` | Whether to run in debug mode or not. |
| `eval` | `e` | Whether to run evaluation mode or in optmization mode. |
| `resume` | `r` | Whether to restore previously saved optimization checkpoint. |
| `ncpu` | `n` | Number of cores to use for parallelization. |
| `cfg` | `f` | JSON configuration file to be used (without extension). The file has to be stored in `spike_swarm_sim/config` |
| `verbose` | `v` | Whether to run in verbose mode. |

## Configuration Files
The configuration file is composed by the following main blocks:
- `checkpoint_file`: name of the file where optimization checkpoints are stored.

- `topology`: configuration of the artificial neural network.

- `algorithm`: configuration of the algorithm that optimizes the previously specified ANN parameters. 

- `world`: configuration of the environment/world.

Firstly, an example of ANN declaration through the `topology` field is shown below. It exemplifies the use of rate neuron 
models as building blocks of CTRNNs:
```json
"topology" : {
    "dt" : 0.1, \\ Euler step of the ANN.
    "time_scale" : 20, \\ Ratio between neuronal and environment time scales (neurons are 20 times faster).
    "stimuli": { \\ Declaration of stimuli fed to the ANN. 
        "W1" : {"n" : 3, "sensor" : "wireless_receiver:msg"}, \\ 3D message of IR communication receiver.
        "W2" : {"n" : 2, "sensor" : "wireless_receiver:receiving_direction"}, \\ 2D message orientation msg.
        "W4" : {"n" : 1, "sensor" : "wireless_receiver:signal"}, \\ Signal strength
        "W6" : {"n" : 6, "sensor" : "light_sensor"} \\ Light sensor (6 sectors).
    },
    \\ Encoding of the above stimuli. Mainly used when spiking neuron models are employed, to map stimulus 
    \\ to spike trains. As it is not the case of this example, identity encoding is fixed as a placeholder.
    \\ In this case, the encoding field could have been completely removed (as it is not used). 
    "encoding" : {
        "W1" :{"scheme" : "IdentityEncoding"},
        "W2" :{"scheme" : "IdentityEncoding"},
        "W4" :{"scheme" : "IdentityEncoding"},
        "W6" :{"scheme" : "IdentityEncoding"}
    },
    \\ Neuron model from spike_swarm_sim.neural_networks.neuron_models.
    "neuron_model" : "rate_model",
    \\ Synapse model from spike_swarm_sim.neural_networks.synapses. Currently, it can be either static for 
    \\ non-spiking neurons and dynamic for spiking neurons.
    "synapse_model" : "static_synapse",
    \\ Set of neuron ensembles or layers. 
    "ensembles": {
        "C1" : {"n" : 10, "params" : {}},
        "C2" : {"n" : 10, "params" : {}},
        "OUT_COMM" : {"n" : 3, "params" : {}},
        "OUT_COMM_ST" : {"n" : 1, "params" : {}},
        "OUT_MOT" : {"n" : 2, "params" : {}}
    },
    \\ Set of ouputs, linking actuators to motor ensembles or neurons.
    "outputs" : {
        \\ OUT_COMM_ST generates the action of the IR communication state (RELAY or SEND).
        "outC" : {"ensemble" : "OUT_COMM_ST", "actuator" : "wireless_transmitter:state", "enc": "cat"},
        \\ OUT_COMM generates the action of the IR communication 3D message.
        "outA" : {"ensemble" : "OUT_COMM", "actuator" : "wireless_transmitter", "enc": "real"},
        \\ OUT_MOT generates the action of the wheel actuator (2D).
        "outB" : {"ensemble" : "OUT_MOT", "actuator" : "wheel_actuator", "enc": "real"}
    },
    \\ Set of ANN synapses, specifying the pre and post synaptic ensembles, the probability of 
    \\ pairwise neuron connection (p) and whether the connection is trainable or not.
    "synapses" :  {
        "W1-c" : {"pre":"W1","post":"C1", "trainable":true, "p":1.0},
        "W2-c" : {"pre":"W2","post":"C1", "trainable":true, "p":1.0},
        "W4-c" : {"pre":"W4","post":"C1", "trainable":true, "p":1.0},
        "W6-c" : {"pre":"W6","post":"C1", "trainable":true, "p":1.0},
        "W1-c1" : {"pre":"W1","post":"C2", "trainable":true, "p":1.0},
        "W2-c1" : {"pre":"W2","post":"C2", "trainable":true, "p":1.0},
        "W4-c1" : {"pre":"W4","post":"C2", "trainable":true, "p":1.0},
        "W6-c1" : {"pre":"W6","post":"C2", "trainable":true, "p":1.0},
        "c-c" : {"pre":"C1","post":"C1", "trainable":true, "p":0.8},
        "c2-c2" : {"pre":"C2","post":"C2", "trainable":true, "p":0.7},
        "c1-c2" : {"pre":"C1","post":"C2", "trainable":true, "p":0.7},
        "c-mot" : {"pre":"C1","post":"OUT_MOT", "trainable":true, "p":1.0},
        "c2-comm" : {"pre":"C2","post":"OUT_COMM", "trainable":true, "p":1.0},
        "c2-st" : {"pre":"C2","post":"OUT_COMM_ST", "trainable":true, "p":1.0},
        "comm-c" : {"pre":"OUT_COMM","post":"C1", "trainable":true, "p":1.0},
        "mot-c" : {"pre":"OUT_MOT","post":"C1", "trainable":true, "p":1.0},
        "st-c" : {"pre":"OUT_COMM_ST","post":"C1", "trainable":true, "p":1.0},
        "comm-comm" : {"pre":"OUT_COMM","post":"OUT_COMM", "trainable":true, "p":1.0},
        "comm-st" : {"pre":"OUT_COMM","post":"OUT_COMM_ST", "trainable":true, "p":1.0},
        "st-comm" : {"pre":"OUT_COMM_ST","post":"OUT_COMM", "trainable":true, "p":1.0}
    },
    \\ Decoding of the output. In this case, it decodes firing rates into actions, but in the 
    \\ case of spiking neurons it maps spike trains into actions. 
    "decoding" : {
        \\ Threshold decoding applies Heaviside mapping of the neurons' output to create binary actions.
        "outC" : {"scheme" : "ThresholdDecoding", "params" : {"is_cat" : true}},
        "outA" : {"scheme" : "IdentityDecoding", "params" : {"is_cat" : false}},
        "outB" : {"scheme" : "IdentityDecoding", "params" : {"is_cat" : false}}
    } 
}
```

An example of `world` configuration is the following:

```json
"world":{
    "world_delay" : 1, \\ Delay of the simulation in visual/render mode.
    "render_connections" :true, \\ Whether to draw an edge when two agents can communicate.
    "height":1000, \\ Height of the world.
    "width": 1000, \\ Width of the world.
    \\ Set of objects to be instantiated.
    "objects" : {
        \\ Robots 
        "robotA" : {
            "type" : "robot", \\ object type
            "num_instances" : 10, \\ number of instances
            "controller" : "neural_controller", \\ Name of the controller. 
            \\ Set of sensors with their parameters (unspecified parameters are autocompleted with defaults). 
            "sensors" : {
                "wireless_receiver" : {"n_sectors":4, "range" : 150,  "msg_length" : 3}, 
                "light_sensor" : {"n_sectors" : 6}
            },
            \\ Set of actuators with their parameters (unspecified parameters are autocompleted with defaults).
            "actuators" : {
                "wheel_actuator" : {}, 
                "wireless_transmitter" : {"quantize":true, "range" : 150, "msg_length" : 3}
            },
            \\ Initialization of robots within the environment.
            "initializers" : {
                "positions" : {"name" : "random_uniform", "params" : {"low":400, "high" : 600, "size" : 2}},
                "orientations" : {"name" : "random_uniform",  "params" : {"low":0, "high" : 6.28, "size" : 1}}
            },
            \\ Perturbations applied to the robots at runtime. In this case light sensor stimuli of 
            \\ 8 out of 10 robots is inhibited, so that only 2 robots can sense the light.
            "perturbations" : {"stimuli_inhibition" : {"affected_robots": 8, "stimuli" : "light_sensor"}},
            \\ Additional parameters.
            "params" : {"trainable" : true}
        },
        \\ Light source
        "light_red" : {
            "type" : "light_source",
            "num_instances": 1,
            "controller" : "light_orbit_controller",
            "positions" : "random",
            "initializers" : {
                "positions" : {"name" : "random_circumference", "params" : {"radius": 1, "center" : [500, 500]}}
            },
            "params" : {"range" : 80, "color" : "red"}
        }
    }
}
```
An example of `algorithm` configuration is the following (using CTRNN):
```json
"algorithm" : {
    "name" : "GA", \\ Name of the algorithm (Genetic Algorithm in this case).
    "evolvable_object" : "robotA", \\ Reference to the entity to be evolved.
    "population_size" : 100, \\ Population size
    "generations" : 1000, \\ Number of generations.
    "evaluation_steps" : 1000, \\ Evaluation steps of each simulation trial.
    "num_evaluations" : 3, \\ Number of trials to estimate the fitness.
    "fitness_function" : "goto_light", \\ Name of the fitness function.
    \\ Set of populations. In this case there is only one population, but 
    \\ multiple population implementing cooperative coevolution are supported.
    "populations" : {
        "p1" : {
            \\ Parts of the ANN to be evolved.
            "objects" : ["synapses:weights:all", "neurons:bias:all",  "neurons:tau:all", "neurons:gain:all"],
            \\ Maximum search space bounds.
            "max_vals" : [3,  2, 1, 4],
            \\ Minimum search space bounds.
            "min_vals" : [-3, -2, -1, 0.5],
            \\ Algorithm dependend parameters
            "params": {
                "encoding" : "real", 
                "selection_operator" : "nonlin_rank",
                "crossover_operator" : "blxalpha",
                "mutation_operator" : "gaussian",
                "mating_operator" : "random",
                "mutation_prob" : 0.05,
                "crossover_prob" : 0.9,
                "num_elite" : 2
            }
        }
    }
} 
```

Note that, in each population, the `objects` field settles the parts of the ANN specified in `topology` to be 
evolved. The parts of the ANN are configured using a query system that is used in the simulator to address certain 
parts of the ANN. Currently implemented queries are: 

| Query | Target | Description |
| :---:  | :---:  | :--- |
| `synapses:weights` | `all` | Weights of all the synapses.|
|  | `synapse_name` | Weights of synapse with name `synapse_name`. |
|  | `sensory` |  Weights of synapses with a sensory presynaptic neurons. |
|  | `hidden` | Weights of synapses with a hidden presynaptic neurons. |
|  | `motor` | Weights of synapses with a motor presynaptic neurons. |
| `neurons:tau` | `all` | Neuron membrane time constants of all neurons (only rate_model). |
| `neurons:gain` | `all` | Neuron gain of all neurons (only rate_model). |
| `neurons:bias` | `all` | Neuron biases of all neurons (only rate_model). |
| `decoding:weights` | `all` | Decoding weights if using LinearPopulationDecoding in spiking neural nets. |
