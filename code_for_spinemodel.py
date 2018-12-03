# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:59:17 2018

@author: Gaurang Mahajan

##################################################################################################################################
This program simulates calcium and plasticity dynamics in a physiologically plausible mathematical model of an ER-bearing CA1 
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

All other kinetic parameters and species concentrations are listed below, under the heading "Setting model parameters".

All dynamical variables are initially set to arbitrary values, and the model is run for 500 sec (in the absence of inputs) 
which ensures that all variables attain their steady-state resting levels. This state is taken as the initial condition 
at the start of the simulations.
 
The coupled ODEs are integrated first for the ER+ spine head, and then for the ER- spine head (which is equivalent to the 
ER+ case, except that n_ip3r and Vmax_serca are set to 0).

The model includes L-VGCC, which by default is ignored (g_vgcc set to 0) for the rdp input patterns.

The parameter 'rho_spines' specifies the effective total strength of co-active synaptic inputs at the dendritic compartment 
(mimicking SC inputs); it is set to zero by default. 

##################################################################################################################################
"""

#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#### Global constants:
rtol = 1e-6
atol = 1e-6 
F = 96485.33 ## Coulomb/mole
Nav = 6.022e23
e = 2.718

##############################################
#### Prompting the user for input parameters:
 
g_NMDAR,n_ip3r,input_pattern,f_input,n_inputs = raw_input("Input gNMDAR (pS), no. of IP3R, input_type, input frequency (Hz), and no. of inputs: ").split(' ')

if input_pattern not in ['rdp','stdp']:
    print 'input_type can only be "rdp" or "stdp"'
    raise SystemExit

if input_pattern == 'stdp': n_bap, tdiff = raw_input("Input no. of BAP per input and spike timing difference (ms): ").split(' ')


##############################################################################################
#### Defining various functions used in model simulation:

##############################################################################################
#### Temporal profile of glutamate availability:

def glu(t,s):
    
    tau_glu = 1e-3 ## sec
    glu_0 = 2.718 * 300 ## uM
        
    if s == 0: 
        return 0
    if s == 1: 
        total = 0
        for tpuff in tpre:    
            if t > tpuff: total += glu_0 * np.exp(-(t-tpuff)/tau_glu) * ((t-tpuff)/tau_glu)
        return total
##############################################################################################

##############################################################################################
#### Voltage profile of BAP at the dendritic compartment:

def u_bpap(t):

    V0 = 67
    total = 0
    for tbp in tpost:
        if t > tbp: total += V0 * (0.7 * np.exp(-(t-tbp)/0.003) + 0.3 * np.exp(-(t-tbp)/0.04))
    return E_L + total
##############################################################################################
    
##############################################################################################    
#### AMPAR conductance profile: 

def I_A(s,u,t):

    if s==0:
        return 0
    else:
        total = 0
        for tpuff in tpre:
            if t>tpuff: total += g_A * (np.exp(-(t-tpuff)/tau_A2) - np.exp(-(t-tpuff)/tau_A1))  
        return total * (u - E_A)
##############################################################################################
        
##############################################################################################     
#### NMDAR conductance profile:

def I_N(s,u,t):

    if s==0:
        return 0
    if s==1:
        total = 0
        for tpuff in tpre:
            if t>tpuff: total += g_N * (np.exp(-(t-tpuff)/tau_N2) - np.exp(-(t-tpuff)/tau_N1))
        return total * (u - E_N) / (1 + 0.28 * np.exp(-0.062 * u))
##############################################################################################
        
##############################################################################################        
#### Plasticitry model, Omega function

def wfun(x):

    U = -beta2*(x - alpha2)
    V = -beta1*(x - alpha1)
    if U>100: U = 100
    if V>100: V = 100        
    return (1.0/(1 + np.exp(U))) - 0.5*(1.0/(1 + np.exp(V)))
##############################################################################################
    
##############################################################################################    
#### Plasticity model, tau function

def wtau(x):

    return P1 + (P2/(P3 + (2*x/(alpha1+alpha2))**P4))
##############################################################################################

#########################################################################################################################       
#### Coupled ODEs describing the ER-bearing spine head, which is resistively coupled to a dendritic compartement via a passive neck:       

def spine_model(x,t):
    
    R_Gq,Gact,IP3,ca_Gact_PLC_PIP2,DAGdegr,PLC_PIP2,DAG,IP3_IP5P,IP3degr,glu_R_Gq,Gbc,ca_PLC,IP3_IP3K_2ca,R,ca_PLC_PIP2,IP3K_2ca,Gact_PLC_PIP2,Gq,IP5P,GaGDP,ca_Gact_PLC,glu_R,IP3K, \
    pH, pL, cbp, Bslow, calb, calb_m1, calb_h1, calb_m2, calb_h2, calb_m1h1, calb_m2h1, calb_m1h2, c1n0, c2n0, c0n1, c0n2, c1n1, c2n1, c1n2, c2n2, mv, hv, w, u, ud,\
    h, ca_er, ca = x

    nt = glu(t,s)

    if s>0 and input_pattern=='stdp': ud = u_bpap(t)
        
    ## mGluR-IP3 pathway:    

    R_Gq_eq = -a2f*R_Gq*nt+a2b*glu_R_Gq+a3f*R*Gq-a3b*R_Gq
    Gact_eq = +a5*glu_R_Gq+a6*Gq-a7*Gact-b3f*Gact*PLC_PIP2+b3b*Gact_PLC_PIP2-b4f*Gact*ca_PLC_PIP2+b4b*ca_Gact_PLC_PIP2-b5f*ca_PLC*Gact+b5b*ca_Gact_PLC
    IP3_eq = +b6*ca_PLC_PIP2+b7*ca_Gact_PLC_PIP2-100*IP3K_2ca*IP3+80*IP3_IP3K_2ca-9*IP5P*IP3+72*IP3_IP5P
    ca_Gact_PLC_PIP2_eq = +b2f*ca*Gact_PLC_PIP2-b2b*ca_Gact_PLC_PIP2+b4f*Gact*ca_PLC_PIP2-b4b*ca_Gact_PLC_PIP2-b11*ca_Gact_PLC_PIP2+b9f*ca_Gact_PLC*PIP2-b9b*ca_Gact_PLC_PIP2-b7*ca_Gact_PLC_PIP2
    DAGdegr_eq = +DAGdegrate*DAG
    PLC_PIP2_eq = -b1f*ca*PLC_PIP2+b1b*ca_PLC_PIP2-b3f*Gact*PLC_PIP2+b3b*Gact_PLC_PIP2+b10*Gact_PLC_PIP2
    DAG_eq = +b6*ca_PLC_PIP2+b7*ca_Gact_PLC_PIP2-DAGdegrate*DAG
    IP3_IP5P_eq = +9*IP5P*IP3-72*IP3_IP5P-18*IP3_IP5P
    IP3degr_eq = +20*IP3_IP3K_2ca+18*IP3_IP5P
    glu_R_Gq_eq = +a2f*R_Gq*nt-a2b*glu_R_Gq+a4f*glu_R*Gq-a4b*glu_R_Gq-a5*glu_R_Gq
    Gbc_eq = +a5*glu_R_Gq+a6*Gq-a8*GaGDP*Gbc
    ca_PLC_eq = -b8f*ca_PLC*PIP2+b8b*ca_PLC_PIP2+b6*ca_PLC_PIP2-b5f*ca_PLC*Gact+b5b*ca_Gact_PLC+b12*ca_Gact_PLC
    IP3_IP3K_2ca_eq = +100*IP3K_2ca*IP3-80*IP3_IP3K_2ca-20*IP3_IP3K_2ca
    R_eq = -a1f*R*nt+a1b*glu_R-a3f*R*Gq+a3b*R_Gq
    ca_PLC_PIP2_eq = +b1f*ca*PLC_PIP2-b1b*ca_PLC_PIP2-b4f*Gact*ca_PLC_PIP2+b4b*ca_Gact_PLC_PIP2+b11*ca_Gact_PLC_PIP2+b8f*ca_PLC*PIP2-b8b*ca_PLC_PIP2-b6*ca_PLC_PIP2
    IP3K_2ca_eq = +1111*IP3K*ca*ca-100*IP3K_2ca-100*IP3K_2ca*IP3+80*IP3_IP3K_2ca+20*IP3_IP3K_2ca
    Gact_PLC_PIP2_eq = -b2f*ca*Gact_PLC_PIP2+b2b*ca_Gact_PLC_PIP2+b3f*Gact*PLC_PIP2-b3b*Gact_PLC_PIP2-b10*Gact_PLC_PIP2
    Gq_eq = -a3f*R*Gq+a3b*R_Gq-a4f*glu_R*Gq+a4b*glu_R_Gq-a6*Gq+a8*GaGDP*Gbc
    IP5P_eq = -9*IP5P*IP3+72*IP3_IP5P+18*IP3_IP5P
    GaGDP_eq = +a7*Gact-a8*GaGDP*Gbc+b10*Gact_PLC_PIP2+b11*ca_Gact_PLC_PIP2+b12*ca_Gact_PLC
    ca_Gact_PLC_eq = -b9f*ca_Gact_PLC*PIP2+b9b*ca_Gact_PLC_PIP2+b7*ca_Gact_PLC_PIP2+b5f*ca_PLC*Gact-b5b*ca_Gact_PLC-b12*ca_Gact_PLC
    glu_R_eq = +a1f*R*nt-a1b*glu_R-a4f*glu_R*Gq+a4b*glu_R_Gq+a5*glu_R_Gq
    IP3K_eq = -1111*IP3K*ca*ca+100*IP3K_2ca
    
    ca_eq = (-b1f*ca*PLC_PIP2-b2f*ca*Gact_PLC_PIP2-1111*IP3K*ca*ca-1111*IP3K*ca*ca + (b1b*ca_PLC_PIP2+b2b*ca_Gact_PLC_PIP2+100*IP3K_2ca+100*IP3K_2ca)) 

    ## IP3 receptor kinetics:

    x = IP3/(IP3 + d1)
    y = ca/(ca + d5)
    Q2 = Kinh
    h_eq = a2*(Q2 - (Q2+ca)*h)
    
    ca_eq += ip3r_tot * ((x*y*h)**3) * alpha_ip3r * (ca_er - ca)/(Nav * Vspine) 

    ca_er_eq = -alpha_ip3r * ip3r_tot * ((x*y*h)**3) * (ca_er - ca)/(Nav * Ver)  +  (ca_er_0 - ca_er)/tau_refill
    
    ## Buffer equations:

    Bslow_eq = -kslow_f*Bslow*ca + kslow_b*(Bslow_tot - Bslow)
    ca_eq += -kslow_f*Bslow*ca + kslow_b*(Bslow_tot - Bslow)
    
    cbp_eq = -kbuff_f*ca*cbp + kbuff_b*(cbp_tot - cbp)
    ca_eq += -kbuff_f*ca*cbp + kbuff_b*(cbp_tot - cbp)    
    
    calb_m2h2 = calb_tot - calb - calb_m1 - calb_h1 - calb_m2 - calb_h2 - calb_m1h1 - calb_m2h1 - calb_m1h2
    calb_eqs = [ -ca*calb*(km0m1 + kh0h1) + km1m0*calb_m1 + kh1h0*calb_h1,\
                     ca*calb*km0m1 - km1m0*calb_m1 + calb_m2*km2m1 - ca*calb_m1*km1m2 + calb_m1h1*kh1h0 - ca*calb_m1*kh0h1,\
                     ca*calb*kh0h1 - kh1h0*calb_h1 + calb_h2*kh2h1 - ca*calb_h1*kh1h2 + calb_m1h1*km1m0 - ca*calb_h1*km0m1,\
                     ca*calb_m1*km1m2 - km2m1*calb_m2 + kh1h0*calb_m2h1 - ca*kh0h1*calb_m2,\
                     ca*calb_h1*kh1h2 - kh2h1*calb_h2 + km1m0*calb_m1h2 - ca*km0m1*calb_h2,\
                     ca*(calb_h1*km0m1 + calb_m1*kh0h1) - (km1m0+kh1h0)*calb_m1h1 - ca*calb_m1h1*(km1m2+kh1h2) + kh2h1*calb_m1h2 + km2m1*calb_m2h1,\
                     ca*km1m2*calb_m1h1 - km2m1*calb_m2h1 + kh2h1*calb_m2h2 - kh1h2*ca*calb_m2h1 + kh0h1*ca*calb_m2 - kh1h0*calb_m2h1,\
                     ca*kh1h2*calb_m1h1 - kh2h1*calb_m1h2 + km2m1*calb_m2h2 - km1m2*ca*calb_m1h2 + km0m1*ca*calb_h2 - km1m0*calb_m1h2 ]
    ca_eq += -ca*(km0m1*(calb+calb_h1+calb_h2) + kh0h1*(calb+calb_m1+calb_m2) + km1m2*(calb_m1+calb_m1h1+calb_m1h2) + kh1h2*(calb_h1+calb_m1h1+calb_m2h1))+\
                km1m0*(calb_m1+calb_m1h1+calb_m1h2) + kh1h0*(calb_h1+calb_m1h1+calb_m2h1) + km2m1*(calb_m2+calb_m2h1+calb_m2h2) + kh2h1*(calb_h2+calb_m1h2+calb_m2h2)
    
    ## Ca2+/calmodulin kinetics:
    
    c0n0 = cam_tot - c1n0 - c2n0 - c0n1 - c0n2 - c1n1 - c2n1 - c1n2 - c2n2
    c1n0_eq = -(k2c_on*ca + k1c_off + k1n_on*ca)*c1n0 + k1c_on*ca*c0n0 + k2c_off*c2n0 + k1n_off*c1n1
    c2n0_eq = -(k2c_off + k1n_on*ca)*c2n0 + k2c_on*ca*c1n0 + k1n_off*c2n1
    c0n1_eq = -(k2n_on*ca + k1n_off + k1c_on*ca)*c0n1 + k1n_on*ca*c0n0 + k2n_off*c0n2 + k1c_off*c1n1
    c0n2_eq = -(k2n_off + k1c_on*ca)*c0n2 + k2n_on*ca*c0n1 + k1c_off*c1n2
    c1n1_eq = -(k2c_on*ca + k1c_off + k1n_off + k2n_on*ca)*c1n1 + k1c_on*ca*c0n1 + k1n_on*ca*c1n0 + k2c_off*c2n1 + k2n_off*c1n2
    c2n1_eq = -(k2c_off + k2n_on*ca)*c2n1 + k2c_on*ca*c1n1 + k2n_off*c2n2 + k1n_on*ca*c2n0 - k1n_off*c2n1
    c1n2_eq = -(k2n_off + k2c_on*ca)*c1n2 + k2n_on*ca*c1n1 + k2c_off*c2n2 + k1c_on*ca*c0n2 - k1c_off*c1n2
    c2n2_eq = -(k2c_off + k2n_off)*c2n2 + k2c_on*ca*c1n2 + k2n_on*ca*c2n1
    cam_eqs = [c1n0_eq, c2n0_eq, c0n1_eq, c0n2_eq, c1n1_eq, c2n1_eq, c1n2_eq, c2n2_eq]
    ca_eq += -ca*(k1c_on*(c0n0+c0n1+c0n2) + k1n_on*(c0n0+c1n0+c2n0) + k2c_on*(c1n0+c1n1+c1n2) + k2n_on*(c0n1+c1n1+c2n1)) + \
    k1c_off*(c1n0+c1n1+c1n2) + k1n_off*(c0n1+c1n1+c2n1) + k2c_off*(c2n0+c2n1+c2n2) + k2n_off*(c0n2+c1n2+c2n2)
 
    ## PMCA/NCX kinetics:
    
    ca_eq += pH*kH_leak - ca*pH*k1H + k2H*(pHtot - pH)  +  pL*kL_leak - ca*pL*k1L + k2L*(pLtot - pL)
    pH_eq = k3H*(pHtot - pH) - ca*pH*k1H + k2H*(pHtot - pH)
    pL_eq = k3L*(pLtot - pL) - ca*pL*k1L + k2L*(pLtot - pL)
    
    
    ## SERCA kinetics:

    ca_eq += -Vmax_serca * ca**2/(Kd_serca**2 + ca**2) + k_erleak*(ca_er - ca)

    ## VGCC equatiosn:

    mv_eq = ((1.0/(1 + np.exp(-(u-um)/kmv))) - mv)/tau_mv
    hv_eq = ((1.0/(1 + np.exp(-(u-uh)/khv))) - hv)/tau_hv
    I_vgcc = -0.001 * Nav * 3.2e-19 * g_vgcc * (mv**2) * hv * 0.078 * u * (ca - ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u))
    
    ## Spine and dendrite voltage eqns:

    sp_hh_eq = -(1/Cmem) * ( g_L*(u - E_L) + I_A(s,u,t)/Aspine + I_N(s,u,t)/Aspine - (gc/Aspine)*(ud - u) - I_vgcc/Aspine)
    dend_hh_eq = -(1/Cmem) * ( g_L*(ud - E_L) + rho_spines*gc*(ud - u))

    ## Ca2+ influx through NMDAR and VGCC:

    ca_eq += -(g_N_Ca/Vspine) * (I_N(s,u,t)/(g_N*(u - E_N))) * 0.078 * u * (ca - ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u)) \
            -(g_vgcc/Vspine) * (mv**2) * hv * 0.078 * u * (ca - ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u))   
    
    ## Equation for plasticity variable w:

    acam = cam_tot - c0n0    
    w_eq = (1.0/wtau(acam))*(wfun(acam) - w)
    
    return [R_Gq_eq,Gact_eq,IP3_eq,ca_Gact_PLC_PIP2_eq,DAGdegr_eq,PLC_PIP2_eq,DAG_eq,IP3_IP5P_eq,IP3degr_eq,glu_R_Gq_eq,Gbc_eq,ca_PLC_eq,IP3_IP3K_2ca_eq,\
            R_eq,ca_PLC_PIP2_eq,IP3K_2ca_eq,Gact_PLC_PIP2_eq,Gq_eq,IP5P_eq,GaGDP_eq,ca_Gact_PLC_eq,glu_R_eq,IP3K_eq]+\
            [pH_eq, pL_eq, cbp_eq, Bslow_eq] + calb_eqs + cam_eqs + [mv_eq, hv_eq] + [w_eq] + [sp_hh_eq, dend_hh_eq,\
            h_eq, ca_er_eq, ca_eq]
##############################################################################################################################################################


##############################################
#### Setting model parameters:
##############################################

## Spine compartment and ER size:
Vspine = 0.06 ## um^3
d_spine = (6*Vspine/3.14)**0.333  ## um
Aspine = 3.14 * d_spine**2 * 1e-8  ## cm^2
Vspine = Vspine * 1e-15 ## liter
Aer = 0.1 * Aspine  ## cm^2
Ver = 0.1 * Vspine  ## liter
Vspine = Vspine-Ver  ## liter


## Reaction parameters for mGluR_IP3 pathway:
PIP2 = 4000  ## uM
a1f = 11.1 ## /uM/s
a1b = 2  ## 100 /s
a2f = 11.1 ## /uM/s
a2b = 2  ## 100 /s
a3f = 2 ## /uM/s
a3b = 100 ## /s
a4f = 2 ## /uM/s
a4b = 100 ## /s
a5 = 116  ## 116 /s
a6 = 0.001 ## /s
a7 = 0.02 ## /s
a8 = 6 ## /s
b1f = 300 ## /uM/s
b1b = 100 ## /s
b2f = 900 ## /uM/s
b2b = 30 ## /s
b3f = 800 ## /uM/s
b3b = 40 ## /s
b4f = 1200 ## /uM/s
b4b = 6 ## /s
b5f = 1200 ## /uM/s
b5b = 6 ## /s
b6 = 2 ## /s
b7 = 160 ## /s
b8f = 1 ## /uM/s
b8b = 170 ## /s
b9f = 1 ## /uM/s
b9b = 170 ## /s
b10 = 8 ## /s
b11 = 2  ## 8 /s
b12 = 8 ## /s
DAGdegrate = 0.15 ## /s

## Parameters for IP3R model (Fink et al., 2000 and Vais et al., 2010):
Kinh = 0.2  ## uM
d1 = 0.8 ## uM
d5 = 0.3 ## uM
a2 = 2.7 ## /uM/s
alpha_ip3r = (0.15/3.2)*(1e7)*(1e6)/500.0  ## /uM/sec

## Parameters for endogenous immobile buffer (CBP): 
kbuff_f = 247 ## /uM/s
kbuff_b = 524 ## /s

## Parameters for endogenous slow buffer:
kslow_f = 24.7 ## /uM/s
kslow_b = 52.4 ## /s

## Parameters for calbindin-Ca2+ kinetics:
km0m1=174 ## /uM/s
km1m2=87 ## /uM/s
km1m0=35.8 ## /s
km2m1=71.6 ## /s
kh0h1=22 ## /uM/s
kh1h2=11 ## /uM/s
kh1h0=2.6 ## /s
kh2h1=5.2 ## /s

## Parameters for PMCA and NCX pumps:
k1H,k2H,k3H,kH_leak = [150,15,12,3.33]  ## (/uM/s, /s, /s, /s)
k1L,k2L,k3L,kL_leak = [300,300,600,10]  ## (/uM/s, /s, /s, /s)

## Parameters for CaM-Ca2+ interaction:
k1c_on = 6.8  ## /uM/s
k1c_off = 68  ## /s
k2c_on = 6.8 ## /uM/s
k2c_off = 10 ## /s
k1n_on = 108 ## /uM/s
k1n_off = 4150 ## /s
k2n_on = 108 ## /uM/s
k2n_off = 800 ## /s

## Membrane and leak parameters:
Cmem = 1e-6 ##  F/cm^2
g_L = 2e-4  ## S/cm^2
E_L = -70   ## mV

## AMPA receptor parameters:
tau_A1 = 0.2e-3 ## s
tau_A2 = 2e-3  ## s
E_A = 0  ## mV
g_A = 0.5e-9  ## S

## NMDA receptor parameters:
tau_N1 = 5e-3 ## s
tau_N2 = 50e-3 ## s
E_N = 0  ## mV
g_N = float(g_NMDAR) * 1e-12  ## S

## L-VGCC parameters:
um = -20 ## mV
kmv = 5  ## mV
tau_mv = 0.08e-3 ## sec
uh = -65  ## mV
khv = -7 ## mV			 
tau_hv = 300e-3  ## sec

## Spine neck parameters:
Rneck = 1e8  ## Ohm
gc = 1.0/Rneck ## S
rho_spines = 0  ## Surface density of co-active synaptic inputs on dendritic compartment (cm^-2)

## SERCA kinetic parameters:
Vmax_serca = 1  ## uM/sec
Kd_serca = 0.2 ## uM

## Parameters for Ca2+-based plasticity model:
P1,P2,P3,P4 = [1.0,10.0,0.001,2]
beta1,beta2 = [60,60]  ## /uM
alpha1,alpha2 = [2.0,20.0] ## uM

## ER refilling timescale:
tau_refill = 1e-6  ## sec


#########################################################
########### Concentrations of various species ###########
#########################################################

## External Ca (uM):
ca_ext = 2e3

## Resting cytosolic Ca (uM):
ca_0 = 0.05

## Resting Ca in ER (uM):
ca_er_0 = 250 

## Total calbindin concentration in spine (uM):
calb_tot = 45

## Total CBP concentration in the spine (uM):
cbp_tot = 80

## Total slow buffer concentration in the spine (uM):
Bslow_tot = 40

## Total concentration of PMCA and NCX pumps in the spine head (uM):
pHtot = (1e14) * 1000 * Aspine/(Nav * Vspine)
pLtot = (1e14) * 140 * Aspine/(Nav * Vspine)

## Total concentration of CaM in spine (uM):
cam_tot = 50

## Total concentrations of IP3 3-kinase and IP3 5-phosphatase in the spine (uM):
ip5pconc = 1
ip3kconc = 0.9

## Number of IP3R:
ip3r_tot = int(float(n_ip3r))

#########################################################################################



###########################################################################################################
#### Start of simulations
###########################################################################################################

##########################################################################################################
#### Initializing all variables:
#########################################################################################################

mGluR_init = [0,0,0,0,0,0.8,0,0,0,0,0,0,0,0.3,0,0,0,1.0,ip5pconc,0,0,0,ip3kconc]
pumps_init = [pHtot, pLtot]
buff_init =  [cbp_tot, Bslow_tot] + [calb_tot,0,0,0,0,0,0,0] 
CaM_init = [0]*8  
vgcc_init = [0,1] 
w_init = [0] 
voltage_init = [E_L, E_L]
ip3r_init = [1]
ca_init = [ca_er_0, ca_0]
     
xinit0 = mGluR_init + pumps_init + buff_init + CaM_init + vgcc_init + w_init + voltage_init + ip3r_init + ca_init

g_N_Ca = 0.1 * (g_N/(2*F*78.0*ca_ext)) * 1e6   ## Ca conductance of NMDAR channels; liter/sec
if input_pattern == 'rdp': g_vgcc = 0  ## VGCC ignored for presynaptic-only inputs
else: g_vgcc = g_N_Ca
k_erleak = Vmax_serca * (ca_0**2)/((Kd_serca**2 + ca_0**2)*(ca_er_0 - ca_0))  ## /s

###################################################################################################################
 
########################################################################################################
#### Running the ER+ spine model in the absence of inputs to converge to steady state (resting state): 
########################################################################################################
  
s = 0      

t0 = np.linspace(0,500,5000)
sol = odeint(spine_model,xinit0,t0,atol=atol,rtol=rtol)

print 'Initial spine Ca = ', round(sol[-1,-1],3),'uM'
print 'Initial ER Ca = ', round(sol[-1,-2],3),'uM'
print 'Initial spine IP3 = ', round(sol[-1,2],3),'uM'

########################################################################################################

###################################################################################
#### Integrating the ER+ spine model in the presence of user-defined input pattern:
###################################################################################

s = 1

tpre = [i/float(f_input) for i in range(int(float(n_inputs)))]

if input_pattern == 'stdp':
    tdiff = 0.001*float(tdiff)
    n_bap = int(float(n_bap))
    if n_bap == 1: tpost = [i + tdiff for i in tpre]
    if n_bap == 2: tpost = [i + tdiff for i in tpre] + [i + tdiff - 0.01 for i in tpre] 
    t = np.linspace(np.min(tpre+tpost),np.max(tpre+tpost)+1.,10000)
    
else: 
    t = np.linspace(np.min(tpre),np.max(tpre)+1.,10000)

xinit = sol[-1,:]
sol = odeint(spine_model,xinit,t,atol=atol,rtol=rtol)

open_prob = [((ip3/(ip3+d1))*(ca/(ca + d5))*h)**3 for ip3,ca,h in zip(sol[:,2],sol[:,-1],sol[:,-3])]
nmdar_flux = [-(g_N_Ca/Vspine) * (I_N(s,u,i)/(g_N*(u - E_N))) * 0.078 * u * (ca - ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u)) for i,u,ca in zip(t,sol[:,-5],sol[:,-1])]
ip3r_flux = [alpha_ip3r * ip3r_tot * op * (ca_er - ca)/(Nav * Vspine) for op,ca_er,ca in zip(open_prob,sol[:,-2],sol[:,-1])]  
############################################################################################################################################################################

#######################################################################################################
#### Simulating the ER- spine model responding to the same input pattern as above:
#######################################################################################################

ip3r_tot = 0
Vmax_serca = 0
k_erleak = 0

s = 0
sol0 = odeint(spine_model,xinit0,t0,atol=atol,rtol=rtol)

s = 1
xinit = sol0[-1,:]
sol0 = odeint(spine_model,xinit,t,atol=atol,rtol=rtol)
########################################################################################################


####################################################################################################
#### Plotting time course of various quantities:
####################################################################################################

plt.subplot(3,2,1)
plt.plot(t,sol0[:,-1],'b-',label='Ca2+ no stores',linewidth=2.2)
plt.plot(t,sol[:,-1],'r-',label='Ca2+ with ER',linewidth=2.2)
plt.legend(loc='best')
plt.ylabel('Concentration ($\mu$M)',size=20)

plt.subplot(3,2,3)
plt.plot(t,[np.sum(sol0[k,34:42]) for k in range(len(t))],'b--',label='aCaM no stores',linewidth=2.2)
plt.plot(t,[np.sum(sol[k,34:42]) for k in range(len(t))],'r--',label='aCaM with ER',linewidth=2.2)
plt.legend(loc='best')
plt.ylabel('Concentration ($\mu$M)',size=20)

plt.subplot(3,2,2)
plt.plot(t,sol[:,-5],'k-',label='Spine V',linewidth=2.5)
plt.legend(loc='best')
plt.ylabel('Voltage (mV)',size=20)

plt.subplot(3,2,4)
plt.plot(t,nmdar_flux,'g-',linewidth=2.5,label='NMDAR flux')
plt.plot(t,ip3r_flux,'m-',linewidth=2.5,label='IP3R flux')
plt.legend(loc='best')
plt.ylabel('Ca influx rate ($\mu$M/s)',size=20)
 
plt.subplot(3,2,5)
plt.plot(t,sol0[:,-6],'b-',label='w no stores',linewidth=2.6)
plt.plot(t,sol[:,-6],'r-',label='w with ER',linewidth=2.6)
plt.legend(loc='best')
plt.ylabel('Synaptic weight change (a.u.)',size=20)
plt.xlabel('Time (s)',size=20)

plt.subplot(3,2,6)
plt.plot(t,sol[:,2],'c-',label='IP3',linewidth=2.2)
plt.legend(loc='best')
plt.xlabel('Time (s)',size=21)
plt.ylabel('Concentration ($\mu$M)',size=20)
plt.xlabel('Time (s)',size=20)

plt.show()            
##################################################################################################           
            
   


