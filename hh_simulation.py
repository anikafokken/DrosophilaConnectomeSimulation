from brian2 import *
import matplotlib.pyplot as plt

start_scope()

# All parameters per meter^2 (SI units)
C_m = 1 * ufarad / meter**2
g_Na = 120 * msiemens / meter**2
g_K = 36 * msiemens / meter**2
g_L = 0.3 * msiemens / meter**2

E_Na = 50 * mV
E_K = -77 * mV
E_L = -54.4 * mV

tau_syn_exc = 5 * ms
tau_syn_inh = 10 * ms
E_syn_exc = 0 * mV
E_syn_inh = -70*mV
g_peak_exc = 0.1 * msiemens / meter**2
g_peak_inh = 0.2 * msiemens / meter**2

I_ext = 10 * uA / meter**2

spike_times = [10, 20, 30]*ms
spike_times = [15, 25, 35]*ms
input_exc = SpikeGeneratorGroup(1, [0, 0, 0], spike_times)
input_inh = SpikeGeneratorGroup(1, [0, 0, 0], spike_times)


eqs = '''
dv/dt = (I_ext
         - g_Na * m**3 * h * (v - E_Na)
         - g_K * n**4 * (v - E_K)
         - g_L * (v - E_L)
         - g_syn_exc * (v - E_syn_exc)
         - g_syn_inh * (v - E_syn_inh)) / C_m : volt

dm/dt = alpha_m * (1 - m) - beta_m * m : 1
dn/dt = alpha_n * (1 - n) - beta_n * n : 1
dh/dt = alpha_h * (1 - h) - beta_h * h : 1

dg_syn_exc/dt = -g_syn_exc / tau_syn_exc : siemens / meter**2
dg_syn_inh/dt = -g_syn_inh / tau_syn_inh : siemens / meter**2

alpha_m = 0.1/mV * (25*mV - v) / (exp((25*mV - v)/(10*mV)) - 1)/ms : Hz
beta_m = 4*exp(-v/(18*mV))/ms : Hz

alpha_h = 0.07*exp(-v/(20*mV))/ms : Hz
beta_h = 1/(exp((30*mV - v)/(10*mV)) + 1)/ms : Hz

alpha_n = 0.01/mV * (10*mV - v) / (exp((10*mV - v)/(10*mV)) - 1)/ms : Hz
beta_n = 0.125*exp(-v/(80*mV))/ms : Hz
'''

G = NeuronGroup(5, eqs, method='exponential_euler', threshold='v > -20*mV', reset='v = E_L')
G.v = E_L
G.m = 0.05
G.h = 0.6
G.n = 0.32
G.g_syn_exc = 0 * msiemens / meter**2
G.g_syn_inh = 0 * msiemens / meter**2


syn_exc = Synapses(input_exc, G, on_pre='g_syn_exc += g_peak_exc')
syn_exc.connect()

syn_inh = Synapses(input_inh, G, on_pre='g_syn_inh += g_peak_inh')
syn_inh.connect()

M = StateMonitor(G, 'v', record=True)

run(50*ms)
for i in range(5):
    plot(M.t/ms, M.v[i]/mV + i*100, label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
