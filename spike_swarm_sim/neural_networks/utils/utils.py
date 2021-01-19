import logging
import inspect
from spike_swarm_sim.utils import merge_dicts, any_duplicates, ConfigException
import spike_swarm_sim.register as reg

#* ---- CONFIGURATION CHECK FUNCTIONS ---- #
def ensembles_checker(ensembles_config, neuron_model):
    if len(ensembles_config) == 0:
        raise ConfigException('No ANN ensemble specified.')
    for name, ens in ensembles_config.items():
        if 'n' in ens and ens['n'] <= 0.:
            raise ConfigException('Ensemble neurons must be greater than 0. '\
                'Error in ensemble {}'.format(name))
        neuron_args = {key : val for key, val in inspect.getfullargspec(\
            reg.neuron_models[neuron_model]).kwonlydefaults.items()}
        if 'params' in ens:
            for p_name, param in ens['params'].items():
                if p_name not in neuron_args:
                    raise ConfigException('The parameter "{}" does not exist for neuron model {} '\
                        'of ensemble {}. Available neuron model arguments are {}.'\
                        .format(p_name, neuron_model, name, tuple(neuron_args.keys())))
                if type(param) != type(neuron_args[p_name]):
                    raise ConfigException(' Wrong type of parameter {} in ensemble {}. Type {} '\
                        'should have been received.'.format(p_name, name, type(neuron_args[p_name]).__name__))

def synapses_checker(topology):
    synapse_config = topology['synapses']
    ensembles = topology['ensembles']
    synapse_params = topology['synapse_params'] if 'synapse_params' in topology else None
    if len(synapse_config) == 0:
        raise ConfigException('No ANN synapses specified.')
    for name, syn in synapse_config.items():
        # Check types
        for var, expected_type in zip(['pre', 'post', 'p', 'ntx_fixed', 'neuroTX', 'trainable'],\
                [str, str, float, bool, str, bool]):
            if var in syn and not isinstance(syn[var], expected_type):
                raise ConfigException('Synapse field "{}" must be of type {} and its type is {}.'\
                    .format(var, expected_type.__name__, type(syn[var]).__name__))
        if syn['pre'] not in merge_dicts([ensembles, topology['stimuli']]):
            raise ConfigException('Presynaptic ensemble "{}" of synapse "{}" does not '\
                'exist in the defined ensembles.'.format(syn['pre'], name))
        if syn['post'] not in ensembles:
            raise ConfigException('Postsynaptic ensemble "{}" of synapse "{}" does not '\
                'exist in the defined ensembles.'.format(syn['post'], name))
        if 'p' in syn and not 0 <= syn['p'] <= 1:
            raise ConfigException('Connection probability of synapse {} must be in the interval [0, 1]'.format(name))
        if 'neuroTX' in syn and syn['neuroTX'] not in ['AMPA', 'GABA', 'NDMA', 'AMPA+NDMA']:
            raise ConfigException('Neurotransmitter {} of synapse {} is not implemented. '\
                'Available neurotransmitters are {}.'.format(syn['neuroTX'], name, ['AMPA', 'GABA', 'NDMA', 'AMPA+NDMA']))
    #* Check no duplicate connections
    if any_duplicates([syn['pre'] + '-' + syn['post'] for syn in synapse_config.values()]):
        raise ConfigException('There are duplicate connections.')
    # if synapse_params is not None:

def encoding_checker(encoding_config):
    pass
def decoding_checker(decoding_config):	
    pass

def neural_net_checker(topology):
    # Check compulsory variables.
    for var in ['stimuli', 'ensembles', 'synapses', 'outputs']:
        if var not in topology or len(topology[var]) == 0:
            raise ConfigException('Required field {} not specified in config. file.'.format(var))
    if 'dt' in topology and topology['dt'] <= 0.:
        raise ConfigException('ANN Euler step must be greater than 0.')
    if 'time_scale' in topology and topology['time_scale'] <= 0.:
        raise ConfigException('ANN time_scale must be greater than 0.')
    if 'neuron_model' in topology and topology['neuron_model'] not in reg.neuron_models:
        raise ConfigException('Neuron model {} not implemented. '\
            'Available neuron models are {}.'.format(topology['neuron_model'], tuple(reg.neuron_models)))
    if 'synapse_model' in topology and topology['synapse_model'] not in reg.synapse_models:
        raise ConfigException('Synapse model {} not implemented. '\
            'Available synapse models are {}.'.format(topology['synapse_model'], tuple(reg.synapse_models)))

    #* --- Check input stimuli config. ---
    for name, stim in topology['stimuli'].items():
        if 'n' in stim and stim['n'] <= 0.:
            raise ConfigException('Input neurons must be greater than 0. '\
                'Error in input {}'.format(name))
        if 'sensor' not in stim:
            raise ConfigException('No associated sensors specified in input {}.'\
                .format(name))
        #TODO Check that sensor exists in robot. It requires association between ANN and robot.
        # if stim['sensor'] not in topology
    #* --- Check output config. ---
    for name, out in topology['outputs'].items():
        if 'ensemble' not in out:
            raise ConfigException('No associated ensemble specified in output {}.'\
                .format(name))
        if out['ensemble'] not in topology['ensembles']:
            raise ConfigException('Specified ensemble {} of output {} does not exist.'\
                .format(out['ensemble'], name))
        #TODO Check that actuator exists in robot. It requires association between ANN and robot.
    ensembles_checker(topology['ensembles'], topology['neuron_model'])
    synapses_checker(topology)


#* ---- CONFIGURATION AUTOCOMPLETION FUNCTIONS ---- #
def autocomplete_ensembles(topology):
    neuron_model = topology['neuron_model']
    for name, ens in topology['ensembles'].items():
        if 'n' not in ens:
            topology['ensembles'][name]['n'] = 10
            logging.warning('The number of neurons of ensemble {} '\
                'was not settled. Fixing 10 neurons.'.format(name))
        if 'params' not in ens:
            topology['ensembles'][name]['params'] = {}
        neuron_args = {key : val for key, val in inspect.getfullargspec(\
                reg.neuron_models[neuron_model]).kwonlydefaults.items()}
        for p_name, default_val in neuron_args.items():
            if p_name not in ens['params']:
                topology['ensembles'][name]['params'][p_name] = default_val
    return topology

def autocomplete_synapses(topology):
    syn_model = topology['synapse_model']
    for name, syn in topology['synapses'].items():
        if 'p' not in syn:
            topology['synapses'][name]['p'] = 1.
            logging.warning('The connection prob. of synapse "{}" '\
                'was not settled. Fixing prob=1.'.format(name))
        for var in ['trainable', 'ntx_fixed']:
            if var not in syn:
                topology['synapses'][name][var] = True
        #TODO: Ampliar a futuros modelos
        if syn_model == 'dynamic_synapse' and 'neuroTX' not in syn:
            topology['synapses'][name]['neuroTX'] = 'AMPA+NDMA'
            logging.warning('No neurotransmitter (neuroTX) of synapse "{}" '\
                'was settled. Fixing neuroTX=AMPA+NDMA.'.format(name))
    #* Autocomplete synapse params.
    default_params = {
        'static_synapse' : {},
        'dynamic_synapse' : {
            "AMPA" : {"gain" : 6., "tau" : 6., "E": None},
            "GABA" : {"gain" : 5., "tau": 10., "E": None},
            "NDMA" : {"gain" : 1., "tau_rise" : 10.0, "tau_decay": 50, "Mg2" : 0.5, "E" : None}
    }}
    if 'synapse_params' not in topology:
        topology['synapse_params'] = default_params[syn_model]
    if syn_model == 'dynamic_synapse':
        for ntx in ['AMPA', 'GABA', 'NDMA']:
            if ntx not in topology['synapse_params']:
                topology['synapse_params'][ntx] = default_params[syn_model][ntx]
            for varname, ntx_var in default_params[syn_model][ntx].items():
                if varname not in topology['synapse_params'][ntx]:
                    topology['synapse_params'][ntx][varname] = ntx_var
    return topology

def autocomplete_neural_net(topology):
    if 'dt' not in topology:
        topology['dt'] = 1.
        logging.warning('No ANN Euler step dt specified. Using dt=1.0')
    if 'time_scale' not in topology:
        topology['time_scale'] = 1
        logging.warning('No ANN time scale specified. Using time_scale=1')
    #TODO autocomplete input dim based on sensor.
    #TODO autocomplete out ensemble dim based on actuator.
    topology = autocomplete_ensembles(topology)
    topology = autocomplete_synapses(topology)
    return topology