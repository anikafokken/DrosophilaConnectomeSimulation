from brian2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

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

total_idxs = 0
previous_idxs = 0

def get_root_id_to_index(neu_array, start_idx):
    mapping = { rid: idx for idx, rid in enumerate(neu_array, start=start_idx)}
    return mapping

print("done")
unique_neurons = pd.unique(df_conn[['pre_root_id', 'post_root_id']].values.ravel())
unique_sugar_neurons = unique_neurons[np.isin(unique_neurons, sugar_GRNs['root_id'].values)]
print("Printing unique_sugar_neurons*********************")
print(unique_sugar_neurons)

print("done")

# Define parameters
trial_num = len(sugar_GRNs['root_id'])
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

def update_index_map(neu_array, root_id_to_index):
    global total_idxs
    print(total_idxs)
    layer_x_map = get_root_id_to_index(neu_array, total_idxs)
    print(f"adding {len(layer_x_map.items())} more neurons")
    print(f"layer map: {layer_x_map}")
    root_id_to_index.update(layer_x_map)
    total_idxs += len(layer_x_map.keys())
    print(total_idxs)

# def update_synapse_idx(neu_array, root_id_to_layer_local):
#     layer_x_map = {
#         rid: (layer_idx, local_idx) for rid, layer_idx, local_idx in root_id_to_layer_local
#     }
#     root_id_to_layer_local.update(layer_x_map)
    

def create_model(conn):
    previous_idxs = 0

    

    root_id_to_index = {}
    root_id_to_layer_local = {}

    sugar_GRN_IDs = sugar_GRNs['root_id']
    print("Printing sugar_GRN_IDs**********************")
    print(sugar_GRN_IDs)

    sugar_G = NeuronGroup(len(sugar_GRN_IDs), 
                          eqs, 
                          threshold='v >= v_thresh', 
                          reset='''
                            v = c
                            u += d
                          ''', 
                          method='rk4')
    # PoissonInput(target=sugar_G, target_var='v', N=len(sugar_G), rate=20*Hz, weight=0.2*mV)
    print("adding sugar GRNs")
    update_index_map(sugar_GRNs['root_id'], root_id_to_index)
    print(root_id_to_index)
    previous_idxs += len(sugar_GRN_IDs)


    
    previous_order_root_ids = sugar_GRN_IDs
    layer_Gs = [sugar_G]
    synapses_list = []
    all_layers = [sugar_G]
    for l in range(2):
        next_order_root_ids = get_n_order_neurons(previous_order_root_ids, df_conn, 2, True, "sugar", True)
        print(f"type of next order IDs: {type(next_order_root_ids)}")
        print(f"type of previous IDs: {type(previous_order_root_ids)}")
        print(f"sugar GRNs: {sugar_GRN_IDs}")
        print(f"next order GRNs: {next_order_root_ids}")
        print(f"adding next order ids for {l} order")
        print(f"length of next order ids: {len(next_order_root_ids)}")

        layer_idx = 0

        # print(previous_order_root_ids)
        # print(next_order_root_ids)
        print(len(root_id_to_index))

        next_order_G = NeuronGroup(
            len(next_order_root_ids),
            eqs,
            threshold='v >= v_thresh', 
            reset='''
            v = c
            u += d
            ''', 
            method='euler')
        
        update_index_map(next_order_root_ids, root_id_to_index)


        print(previous_order_root_ids)
        print(next_order_root_ids)
        edges = conn[
            (conn['pre_root_id'].isin(previous_order_root_ids)) &
            (conn['post_root_id'].isin(next_order_root_ids))
        ]

    #    print(conn['pre_root_id'].dtype)
    #     print(conn['pre_root_id'].dtype)
    #     print(type(previous_order_root_ids))
    #     print(type(next_order_root_ids))
    #     print(edges[['pre_root_id', 'post_root_id']].head()) 
        

        # next_order_G = NeuronGroup(len(next_order_root_ids), eqs, threshold='rand() < p', reset='v=-65*mV', method='euler')
        layer_Gs.append(next_order_G)
        syn = Synapses(layer_Gs[l], layer_Gs[l+1], 
                       model='w : volt',
                       on_pre='v_post += w')
        print(len(layer_Gs[l]), len(layer_Gs[l+1]))
        
        synapses_list.append(syn)

        all_layers.append(next_order_root_ids)

        layer_offsets = []
        offset = 0
        for layer_neuron_ids in all_layers:
            layer_offsets.append(offset)
            offset += len(layer_neuron_ids)

        # pre_group = layer_Gs[i]
        # print(pre_group)
        # post_group = layer_Gs[i+1]
        # print(post_group)

        # N = len(root_id_to_index)
        # print(N)
        # post_size = len(post_group)
        # cutoff = N - post_size

        # post_indices_global = np.array([int(root_id_to_index[rid]) for rid in edges['post_root_id']], dtype=int)


        # mask = post_indices_global >= cutoff
        
        # filtered_edges = edges[mask].reset_index(drop=True)
        # print(filtered_edges)
        # print(f"Indexes before pre and post initialization: {root_id_to_index}")
        # filtered_pre_subset = filtered_edges['pre_root_id'].map(root_id_to_index)
        # filtered_post_subset = filtered_edges['post_root_id'].map(root_id_to_index)
        
        # print(filtered_pre_subset)

        # pre_indices_global = np.array(filtered_pre_subset.tolist())

        # pre_indices = []
        # post_indices = []
        # for rid in sugar_GRN_IDs:
        #     pre_indices.append(root_id_to_index[rid])
        # for rid in next_order_root_ids:
        #     post_indices.append(root_id_to_index[rid])

        # print(pre_indices)
        # print(post_indices)

        pre_offset = layer_offsets[0]  # where source layer starts globally
        post_offset = layer_offsets[1]  # where target layer starts globally

        pre_indices_global = [root_id_to_index[rid] for rid in edges['pre_root_id']]
        post_indices_global = [root_id_to_index[rid] for rid in edges['post_root_id']]

        # Map back to local:
        pre_indices_local = [g - pre_offset for g in pre_indices_global]
        post_indices_local = [g - post_offset for g in post_indices_global]

        # print(pre_indices_global)
        # post_indices_global = np.array(filtered_post_subset.tolist())
        # print(post_indices_global)
        # filtered_post_root_ids = edges['post_root_id'][mask].reset_index(drop=True).to_numpy()

        # pre_start = N - len(pre_group) - len(pre_group)
        # post_start = N - len(post_group)

        print(f"Printing indicies: ")
        # pre_indices_local = pre_indices_global
        # # print(pre_indices)
        # post_indices_local = post_indices_global - post_start
        print("***Pre:", pre_indices_local)
        print("***Post:", post_indices_local)

        # print(f"Pre root:", filtered_pre_root_ids)
        # print("Post root:", filtered_post_root_ids)
        # print(type(filtered_post_root_ids))

        print(f"Presynaptic group size: {len(layer_Gs[l])}")
        print(f"Postsynaptic group size: {len(layer_Gs[l+1])}")
        # print(f"Max pre index: {pre_indices.max()}")
        print(f"Max post index: {max(post_indices_local)}")

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

        print(root_id_to_index)
        print(len(pre_indices_local))
        print(post_indices_local)
        syn.connect(
            i=pre_indices_local, 
            j=post_indices_local)
        syn.w = 15 * mV
        print("done3.75")

        syn.delay = 1*ms # TODO: should this be different based on the neurons

        for syn in synapses_list:
            print(f"Synapses: {len(syn.i)} connections")
            print(f"Weights sample: {syn.w[:5]}")

        previous_order_root_ids = pd.DataFrame(list(next_order_root_ids), columns=['root_id'])['root_id']
        previous_idxs += len(previous_order_root_ids)


    print(sugar_G.v.shape)
    print(type(sugar_G.v))

    # Intial conditions
    # sugar_G.v = -65*mV
    # sugar_G.u = b * sugar_G.v
    # # G.I = '10 * mV/ms' # constant input

    for G in layer_Gs:
        G.v = c
        G.u = b * G.v
        G.I = 10 * mV/ms
    
    layer_Gs[0].I[0] = 200 * mV/ms

    # G.I = '10 * mV/ms' # constant input

    # Monitors
    spikemon = SpikeMonitor(sugar_G)
    sugar_M = StateMonitor(sugar_G, 'v', record=True)

    return layer_Gs, synapses_list, spikemon, sugar_M, root_id_to_index

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
    layer_Gs, synapses_list, spikemon, sugar_M, root_id_to_index = create_model(df_conn)
    print("done3")
    
    synapse_map = create_neuronal_map(synapses_list, root_id_to_index)
    print("Synapse map:", synapse_map)
    pathways = trace_pathways(synapse_map, 0, root_id_to_index)
    print("Pathways", pathways)

    state_monitors = []
    for g in layer_Gs:
        M = StateMonitor(g, 'v', record=True)
        state_monitors.append(M)

    spike_monitors = []
    for g in layer_Gs:
        sm = SpikeMonitor(g, 'v', record=True)
        spike_monitors.append(sm)
    
    print("Layer 1 --> Layer 2: ", synapses_list[0].i)
    print("Weights: ", synapses_list[0].w)

    # Testing
    # print(root_id_to_layer_and_local_idx)
    layer_idx = 1

    rid = get_rid_from_idx(root_id_to_index, 58)
    print(f"type: {type(rid)}")
    print(root_id_to_index[rid])
    local_idx = root_id_to_index[rid]

    print(root_id_to_index[rid])
    layer_Gs[layer_idx].I[root_id_to_index[rid]] = 500 * mV/ms
    print(layer_Gs[layer_idx].I[local_idx])
    
    net = Network(layer_Gs + synapses_list + spike_monitors + state_monitors)
    print("done3.85")
    net.run(300*ms)
    print("done4")
    # Model synaptic dynamics
    # Implement plasticity rules
    # Specify delays
    # Create a Network object

    for i, sm in enumerate(spike_monitors):
        print(f"Layer {i} spikes:", sm.count)

    # Plot results
    figure(figsize=(12, 12))

    # monitors0to2 = [state_monitors[0], state_monitors[1], state_monitors[2]]

    subplot(2,1,1)
    # for key in list(pathways)[:3]:
    layer_idx = 0
    
    # M = state_monitors[0]

    # rid = get_rid_from_idx(root_id_to_index, 0)

    # plt.plot(M.t/ms, M.v[0]/mV, label=f"Path {0} Neuron {0} Neuron: {rid}")
    # rid = get_rid_from_idx(root_id_to_index, 22)

    # M = state_monitors[1]
    # plt.plot(M.t/ms, M.v[58]/mV, label=f"Path {0} Neuron {58} Neuron: {rid}", alpha=0.7, linewidth=1.0)

    for neuron_idx in pathways[2]:
        print(pathways[2])
        print(neuron_idx)
        print(state_monitors)

        M = state_monitors[layer_idx]
        rid = get_rid_from_idx(root_id_to_index, neuron_idx)
        local_idx = neuron_idx
        
        if layer_idx < len(M.v):
            plt.plot(
                M.t/ms, 
                M.v[neuron_idx]/mV, 
                label=f'Path {0} Neuron {neuron_idx} (Neuron: {rid})',
                alpha=0.7,
                linewidth=1.0)
        
        layer_idx += 1
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)')
    plt.title('Membrane potential v')

    # subplot(2,1,2)
    # plot(sugar_M.t/ms, sugar_M.u[0]/(mV/ms))
    # xlabel('Time (ms)')
    # ylabel('u (mV/ms)')
    # title('Recovery variable u')
    # print("done5")
    plt.legend()

def create_neuronal_map(synapses_list, root_id_to_index):
    print("starting creation")
    synapse_map = {}
    for synapse in synapses_list:
        print(synapse.i, synapse.j)
        for pre_idx, post_idx in zip(synapse.i, synapse.j):
            synapse_map[int(pre_idx)] = synapse_map.get(int(pre_idx), []) + [int(post_idx)]
    print(synapse_map)
    for num in synapse_map:
        print(num)
        print(synapse_map[num][0])
        print(f"{get_rid_from_idx(root_id_to_index, num)} -> {get_rid_from_idx(root_id_to_index, synapse_map[num][0])}") # get_rid_from_idx works well
        print(root_id_to_index[get_rid_from_idx(root_id_to_index, num)])
    return synapse_map

def trace_pathways(synapse_map, start_idx, rid_dict, max_depth=5, max_paths=100):
    print("starting tracing")
    pathways = {}
    path_idx = 0
    stack = [[start_idx]]
    print("created variables")

    while stack:
        current_path = stack.pop()
        last_node = current_path[-1]

        # print("new path: ", current_path, get_rid_from_idx(rid_dict, current_path[0]))

        # print(f"[TRACE] Current path: {current_path}")
        # print(f"[TRACE] last_node: {last_node}")

        # If you have layer + local idx in path: unpack properly:
        for layer_idx in range(len(current_path)):
            # print(current_path)
            # print(layer_idx)
            local_idx = current_path[layer_idx]
            # print(f"[TRACE] layer_idx={layer_idx}, local_idx={local_idx}")

            # 🟢 Add this BEFORE calling get_rid_from_idx:
            # print(f"[TRACE] About to call get_rid_from_idx with layer_idx={layer_idx}, local_idx={local_idx}")
        
        if len(current_path) >= max_depth:
            pathways[path_idx] = current_path
            path_idx += 1
            continue
        next_nodes = synapse_map.get(last_node, [])
        
        if not next_nodes:
            pathways[path_idx] = current_path
            path_idx += 1
        else:
            for n in next_nodes:
                if n in current_path:
                    continue
                stack.append(current_path + [n])
        
        if path_idx >= max_paths:
            print(f"Stopped early: hit max_paths = {max_paths}")
            break

        # next_idxs = [synapse_map[current]]
        # if len(next_idxs) > 1:
        #     for idx in next_idxs[1:]:
        #         on_hold.append(idx)
        # elif len(next_idxs) < 1:
        #     current = on_hold[0]
        #     on_hold[0].pop(0)

        # else:
        #     current = next_idxs[0]
        #     next_idxs.pop(0)
    print(pathways)
    return pathways

def get_rid_from_idx(rid_dict, idx):
        matches = [key for key, val in rid_dict.items() if (val == idx)]
        if not matches:
            print(f" No match for layer_idx={idx}, local_idx={idx} in rid dict!")
            print(f"Available keys:: {list(rid_dict.items())[:10]}")
            raise ValueError(f"Cannot find rid for ({idx}, {idx})")
        return matches[0]    
    
run_experiment()   

plt.show()
        



