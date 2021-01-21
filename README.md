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

``` 
python main.py --cfg experiment_configuration -Rv 
```

| Argument | Abbreviation | Description |
| :---:  | :---:  | :--- |
| `render` | `R` | Whether to run in visual or console mode. |
| `debug` | `d` | Whether to run in debug mode or not. |
| `eval` | `e` | Whether to run evaluation mode or in optmization mode. |
| `resume` | `r` | Whether to restore previously saved optimization checkpoint. |
| `ncpu` | `n` | Number of cores to use for parallelization. |
| `cfg` | `f` | JSON configuration file to be used (without extension). The file has to be stored at spike_swarm_sim/config |
| `verbose` | `v` | Whether to run in verbose mode. |

## Configuration Files

