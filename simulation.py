from brian2 import *

start_scope()

N = 100
tau = 10*ms
v0_max = 3.
duration = 1000*ms
eqs = '''
dv/dt = (v0-v)/tau : 1 (unless refractory)
v0 : 1
'''
# Resets when reaching spike of 0.8
G = NeuronGroup(N, eqs, threshold='v>1', reset='v = 0', refractory=5*ms, method='exact') # Number of neurons, defining differential equations
M = SpikeMonitor(G)

G.v = 'i*v0_max/(N-1)'

# statemon = StateMonitor(G, "v", record=0)

# G.v = 5 # initial value

# spikemon = SpikeMonitor(G)

run(duration)

# print('Spike times: %s' % spikemon.t[:])
# plot(statemon.t/ms, statemon.v[0])

# Raster plot
# plot(spikemon.t/ms, spikemon.i, '.k')
# for t in spikemon.t:
#     axvline(t/ms, ls='--', c='C1', lw=3) # axvline draws a vertical line at each recorded spike time

figure(figsize=(12,4))
subplot(121)
plot(M.t/ms, M.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
subplot(122)
plot(G.v0, M.count/duration)
xlabel('v0')
ylabel('Firing rate (sp/s)');
show()