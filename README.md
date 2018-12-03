# ca1-spine-with-er

The program code_for_spinemodel.py simulates calcium and plasticity dynamics in a physiologically plausible mathematical model of an ER-bearing CA1 
dendritic spine head, as described in Mahajan & Nadkarni, bioRxiv 460568 (2018). The following python modules are imported: 
numpy, scipy and matplotlib. The deterministic, coupled ODEs are numerically integrated using the "odeint" function in SciPy.
 
The code prompts for the following inputs (inputs to be entered together in command line and separated by single spaces):
gNMDAR:       NMDAR conductance parameter (pS),
n_ip3r:       Number of IP3 receptors,
input_type:   Either 'rdp' (presynaptic spikes only) or 'stdp' (paired pre and postsynaptic spiking),
f_input:      Input frequency (Hz), 
n_inputs:     Number of  inputs.

If input_type is chosen as 'stdp', the user is prompted for two more inputs (to be entered together, separated by a space):
n_bap:   Number of BAP (1 or 2) paired with every presynaptic spike (BAPs in a pair are separated by 10 ms),
tdiff:   Spike timing difference between presynaptic spike and last postsynaptic BAP (in ms); pre-before-post is
         associated with positive tdiff.

Example usage: 
>>>python code_for_spinemodel.py
"Input gNMDAR (pS), no. of IP3R, input_type, input frequency (Hz), and no. of inputs: "
65 30 stdp 5 5
"Input no. of BAP per input and spike timing difference (ms): "
2 10

All other kinetic parameters and species concentrations are listed in the code, under the heading "Setting model parameters".

All dynamical variables are initially set to arbitrary values, and the model is run for 500 sec (in the absence of inputs) 
which ensures that all variables attain their steady-state resting levels. This state is taken as the initial condition 
at the start of the simulations.
 
The coupled ODEs are integrated first for the ER+ spine head, and then for the ER- spine head (which is equivalent to the 
ER+ case, except that n_ip3r and Vmax_serca are set to 0).

The model includes L-VGCC, which by default is ignored (g_vgcc set to 0) for the rdp input patterns.

The parameter 'rho_spines' specifies the effective total strength of co-active synaptic inputs at the dendritic compartment 
(mimicking SC inputs); it is set to zero by default. 
