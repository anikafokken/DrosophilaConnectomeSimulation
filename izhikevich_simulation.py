from brian2 import *
import pandas as pd
import numpy as np

start_scope()

df_conn = pd.read_csv("database/connections.csv")
df_clas = pd.read_csv("database/classification.csv")
sugar_GRNs = pd.read_csv("database/sugar_GRNs.csv")
bitter_GRNs = pd.read_csv("database/bitter_GRNs.csv")
lowsalt_GRNs = pd.read_csv("database/lowsalt_GRNs.csv")
water_GRNs = pd.read_csv("database/water_GRNs.csv")
sugar_2Ns = pd.read_csv("output_csvs/sugar_2Ns.csv")
bitter_2Ns = pd.read_csv("output_csvs/bitter_2Ns.csv")
lowsalt_2Ns = pd.read_csv("output_csvs/lowsalt_2Ns.csv")
water_2Ns = pd.read_csv("output_csvs/water_2Ns.csv")

print("done")
unique_neurons = pd.unique(df_conn[['pre_root_id', 'post_root_id']].values.ravel())
unique_sugar_neurons = unique_neurons[np.isin(unique_neurons, sugar_GRNs['root_id'].values)]
print("Printing unique_sugar_neurons*********************")
print(unique_sugar_neurons)

def get_root_id_to_index(neu_array):
    root_id_to_index = { rid: idx for idx, rid in enumerate(neu_array)}
    return root_id_to_index

root_id_to_index = get_root_id_to_index(sugar_GRNs['root_id'])
print("done")
# Define parameters
trial_num = len(root_id_to_index)
a = 0.02/ms # time scale of recovery variable u
b = 0.2/second # sensitivity of u to v
c = -65*mV # after-spike reset of v
d = 2*mV/ms # increment of u after spike
v_thresh = 30*mV
w_syn = 0.1*mV
orders = 5

# Specify differential equations
eqs = '''
dv/dt = (0.04/ms/mV)*v**2 + (5/ms)*v + 140/ms*mV - u + I : volt
du/dt = a*(b*v - u) : volt/second
I : volt/second
p : 1
'''

# Define spiking behavior

def create_model(conn):

    # Work with NeuronGroups
    # G = NeuronGroup(trial_num, eqs, 
    #                 threshold='v>=v_thresh', 
    #                 reset='''
    #                 v = c
    #                 u += d
    #                 ''',
    #                 method=euler)

    sugar_GRN_IDs = sugar_GRNs['root_id']
    print("Printing sugar_GRN_IDs**********************")
    print(sugar_GRN_IDs)
    root_id_to_index = {rid: idx for idx, rid in enumerate(sugar_GRN_IDs)}

    sugar_G = NeuronGroup(len(sugar_GRN_IDs), eqs, threshold='rand() < p', reset='v=-65*mV', method='euler')
    PoissonInput(target=sugar_G, target_var='v', N=len(sugar_G), rate=20*Hz, weight=0.2*mV)
    root_id_to_local_index_previous = {rid: idx for idx, rid in enumerate(sugar_GRN_IDs)}

    
    previous_order_root_ids = set(sugar_GRNs['root_id'])
    layer_Gs = [sugar_G]
    synapses_list = []
    for i in range(orders-2):
        next_order_root_ids = get_n_order_neurons(sugar_GRN_IDs, conn, i+2, merge=True, taste="sugar", filtering_on=True)
        root_id_to_local_index_next = {rid: idx for idx, rid in enumerate(list(next_order_root_ids))}

        # print(previous_order_root_ids)
        # print(next_order_root_ids)
        root_id_to_index.update({rid: idx for idx, rid in enumerate(next_order_root_ids, start=len(root_id_to_index))})
        next_order_G = NeuronGroup(
            len(get_n_order_neurons(sugar_GRN_IDs,
            conn, 
            i+2, 
            merge=True, 
            taste="sugar", 
            filtering_on=True)),
              eqs,
              threshold='rand() < p', 
              reset='v=-65*mV', 
              method='euler')
        
        edges = conn[
            (conn['pre_root_id'].isin(previous_order_root_ids)) &
            (conn['post_root_id'].isin(next_order_root_ids))
        ]

        print(conn['pre_root_id'].dtype)
        print(conn['pre_root_id'].dtype)
        print(type(previous_order_root_ids))
        print(type(next_order_root_ids))
        print(edges[['pre_root_id', 'post_root_id']].head())
        

        # next_order_G = NeuronGroup(len(next_order_root_ids), eqs, threshold='rand() < p', reset='v=-65*mV', method='euler')
        layer_Gs.append(next_order_G)
        syn = Synapses(layer_Gs[i], layer_Gs[i+1], 
                       model='w : volt',
                       on_pre='v_post += w')
        
        synapses_list.append(syn)

        pre_indices = np.array([int(root_id_to_local_index_previous[rid]) for rid in edges['pre_root_id']], dtype=int)
        post_indices = np.array([int(root_id_to_local_index_next[rid]) for rid in edges['post_root_id']], dtype=int)
        print(pre_indices)
        print(post_indices)

        nt_array = edges['nt_type'].values

        filtered_weights = np.where(
            np.isin(nt_array, ['ACH', 'GLUT']),
            5,
            np.where(
                nt_array == 'GABA',
                -5,
                0 # TODO: fix this to include other NTs
                )
        )

        syn.connect(
            i=pre_indices, 
            j=post_indices)
        syn.w = filtered_weights * mV
        print("done3.75")

        syn.delay = 1*ms # TODO: should this be different based on the neurons

        previous_order_root_ids = next_order_root_ids
        root_id_to_local_index_previous = root_id_to_local_index_next

    drive = np.zeros(trial_num) * mV/ms
    drive[0] = 10 * mV/ms  # Only neuron 0 gets input

    print(sugar_G.v.shape)
    print(type(sugar_G.v))

    # Intial conditions
    sugar_G.v = -65*mV
    sugar_G.u = b * sugar_G.v
    # G.I = '10 * mV/ms' # constant input
    sugar_G.I = drive


    # Synapses
    # S = Synapses(G, G, '''
    #              w : volt
    #              ''',
    #              on_pre='v_post += w')
    
    # print("done3.25")
    # i_indices = conn['pre_root_id'].map(root_id_to_index).values
    # j_indices = conn['post_root_id'].map(root_id_to_index).values

    # # Connect synapses
    # S.connect(i=i_indices,
    #           j=j_indices)
    # print("done3.5")

    

    # weights = weights * mV
    
    # S.w = weights
    # print("done3.75")

    # S.delay = 1*ms # TODO: should this be different based on the neurons

    # # PoissonGroups TODO: get rid of this?
    # sweet_stimulus = TimedArray([0, 30]*Hz, dt=500*ms)
    # sweet_PG = PoissonGroup(len(sugar_GRNs), rates='sweet_stimulus(t)')

    # Monitors
    spikemon = SpikeMonitor(sugar_G)
    sugar_M = StateMonitor(sugar_G, ['v', 'u'], record=True)

    return layer_Gs, synapses_list, spikemon, sugar_M

# Finds all neurons of a given order (2 or 3)
def get_n_order_neurons(neu_list, connections, order, merge, taste, filtering_on):
    sub_class_terms = {
                "sugar": "sugar/water",
                "bitter": "bitter",
                "ir94e": "low-salt",
                "water": "sugar/water"
            }
    twoNs_dfs = {
                "sugar": sugar_2Ns,
                "bitter": bitter_2Ns,
                "ir94e": lowsalt_2Ns,
                "water": water_2Ns
            }
    
    # Apply synapse threshold (5 for 2Ns, 10 for 3Ns)
    if order == 2:
        connections = connections[connections['syn_count'] >= 5]
    elif order == 3:
        connections = connections[connections['syn_count'] >= 10]

    # Merge GRN list with connections if needed (some input lists may be from connections CSVs, so they don't need to merge with the connections CSV)
    if merge:
        neu_list = pd.merge(neu_list, connections, left_on='root_id', right_on='pre_root_id')

    # Define a 'visited' and 'frontier' list to keep track of upstream neurons that are already accounted for
    current_ids = set(neu_list['root_id'])
    visited = set(current_ids)

    for i in range(order-1):
    #     # Where the pre root ID is the same as the root ID from the input neuron list
        current_conns = connections[connections['pre_root_id'].isin(current_ids)]
        next_ids = set(current_conns['post_root_id'])

        if filtering_on:
            if i == (order-2) and order == 3:
                
                # Get GRNs (gustatory + same taste type)
                grn_filter = (
                    (df_clas["class"] == "gustatory") |
                    (df_clas["root_id"].isin(twoNs_dfs[taste])) # TODO: is this the problem? they filter by no GRNS of any modality, and no 2Ns of the same modality; issue is likely ir94e
                )
                grn_ids = set(df_clas[grn_filter]["root_id"])
                next_ids -= grn_ids

        # Remove already visited; set the current IDs to the ones discovered in this loop, so they can be explored for 3Ns if applicable
        next_ids -= visited
        visited.update(next_ids)
        current_ids = next_ids
    
    return current_ids

def run_experiment():
    G, S, spikemon, M = create_model(df_conn)
    

print("done3")
sugar_G, syn, spikemon, sugar_M = create_model(df_conn)
net = Network([sugar_G, syn, spikemon, sugar_M])
print("done3.85")
net.run(75*ms)
print("done4")
# Model synaptic dynamics
# Implement plasticity rules
# Specify delays
# Create a Network object

# Plot results
figure(figsize=(12, 6))

subplot(2,1,1)
for i in range(3):
    plot(sugar_M.t/ms, sugar_M.v[i]/mV, label=f'Neuron {i}')
xlabel('Time (ms)')
ylabel('v (mV)')
title('Membrane potential v')

# subplot(2,1,2)
# plot(sugar_M.t/ms, sugar_M.u[0]/(mV/ms))
# xlabel('Time (ms)')
# ylabel('u (mV/ms)')
# title('Recovery variable u')
# print("done5")
legend()
show()