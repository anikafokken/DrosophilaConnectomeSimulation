from brian2 import *
import pandas as pd

start_scope()

df_conn = pd.read_csv("database/connections.csv")

# Define parameters
trial_num = 100
a = 0.1/ms # time scale of recovery variable u
b = 0.26/ms # sensitivity of u to v
c = -65*mV # after-spike reset of v
d = 2*mV/ms # increment of u after spike
v_thresh = 30*mV

# Specify differential equations
eqs = '''
dv/dt = (0.04/ms/mV)*v**2 + (5/ms)*v + 140/ms*mV - u + I : volt
du/dt = a*(b*v - u) : volt/second
I : volt/second
'''

# Define spiking behavior

def create_model(conn):
    # Work with NeuronGroups
    G = NeuronGroup(trial_num, eqs, 
                    threshold='v>=v_thresh', 
                    reset='''
                    v = c
                    u += d
                    ''',
                    method=euler)

    # Intial conditions
    G.v = -65*mV
    G.u = b * G.v
    G.I = 10 * mV/ms # constant input

    # Synapses
    S = Synapses(G, G, on_pre='v_post += 5*mV')
    S.connect(G, G, 'w : volt', on_pre='g += w')

    i_pre = conn["pre_root_id"]
    i_post = conn["post_root_id"]
    S.connect(i=i_pre, j=i_post)

    nt = conn["nt_type"]
    excitatory_neus = conn[conn['nt_type'] == 'ACH' or conn['nt_type'] == 'GABA']
    inhibitory_neus = conn[conn['nt_type'] == 'GLUT']

    # Monitors
    spikemon = SpikeMonitor(G)
    M = StateMonitor(G, ['v', 'u'], record=True)

run(200*ms)

# Model synaptic dynamics
# Implement plasticity rules
# Specify delays
# Create a Network object

# Plot results
figure(figsize=(12, 6))

subplot(2,1,1)
plot(M.t/ms, M.v[0]/mV)
xlabel('Time (ms)')
ylabel('v (mV)')
title('Membrane potential v')

subplot(2,1,2)
plot(M.t/ms, M.u[0]/(mV/ms))
xlabel('Time (ms)')
ylabel('u (mV/ms)')
title('Recovery variable u')

show()