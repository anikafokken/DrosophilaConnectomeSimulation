from brian2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from collections import Counter

start_scope()

df_conn = pd.read_csv("database/connections.csv")
df_class = pd.read_csv("database/classification.csv")
sugar_GRNs = pd.read_csv("database/sugar_GRNs.csv")
bitter_GRNs = pd.read_csv("database/bitter_GRNs.csv")
lowsalt_GRNs = pd.read_csv("database/lowsalt_GRNs.csv")
ir94e_GRNs = pd.read_csv("database/ir94e_GRNs.csv")
water_GRNs = pd.read_csv("database/water_GRNs.csv")
sugar_2Ns = pd.read_csv("output_csvs/sugar_2Ns.csv")
bitter_2Ns = pd.read_csv("output_csvs/bitter_2Ns.csv")
lowsalt_2Ns = pd.read_csv("output_csvs/lowsalt_2Ns.csv")
water_2Ns = pd.read_csv("output_csvs/water_2Ns.csv")

total_idxs = 0
post_ids = 0
pre_offset = 0
previous_idxs = 0

def get_root_id_to_index(neu_array, start_idx, seen_dict):
    unique_ids = get_unique_ids(neu_array, seen_dict)
    # print(f"unique IDs: {unique_ids}")
    mapping = { rid: idx for idx, rid in enumerate(unique_ids, start=start_idx)}
    # print(f"mapping: {mapping}")
    return mapping

def get_unique_ids(neu_array, seen_dict):
    unique_ids = []
    for rid in neu_array:
        if rid not in seen_dict:
            unique_ids.append(rid)
            seen_dict[rid] = True
            # print(f"appended {rid}")
        # else:
        #     # print(f"[DUPLICATE] {rid}")
    return unique_ids

def get_rid_from_global_idx(rid_dict, idx):
    matches = [key for key, val in rid_dict.items() if (val == idx)]
    if not matches:
        print(f" No match for local_idx={idx} in rid dict!")
        print(len(rid_dict))
        print(f"Available keys:: {list(rid_dict.items())[:10]}")
        raise ValueError(f"Cannot find rid for ({idx}, {idx})")
    return matches[0]

def get_global_idx_from_layer_local(rid_dict, layer_idx, local_idx):
    matches = [key for key, val in rid_dict.items() if (val == (layer_idx, local_idx))]
    if not matches:
        print(f" No match for local_idx={local_idx} in rid dict!")
        print(len(rid_dict))
        print(f"Available keys:: {list(rid_dict.items())[:10]}")
        raise ValueError(f"Cannot find rid for ({layer_idx}, {local_idx})")
    return matches[0]

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
v_thresh = 30 * mV
w_syn = 5 * mV
orders = 5

neuron_params = {
    "DA": {"a": 0.02, "b": 0.2,  "c": -55, "d": 4},     # tonic/bursting
    "OCT": {"a": 0.1, "b": 0.2, "c": -65, "d": 2},     # fast spiking
    "sugar": {"a": 0.02, "b": 0.25, "c": -65, "d": 2}, # phasic
    "bitter": {"a": 0.02, "b": 0.25, "c": -50, "d": 2}, # bursting
    "Ir94e": {"a": 0.02, "b": 0.2, "c": -65, "d": 6} # regular spiking
}

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
    global post_ids
    print(total_idxs)
    layer_x_map = get_root_id_to_index(neu_array, total_idxs, root_id_to_index)
    print(f"adding {len(layer_x_map.items())} more neurons")
    post_ids = len(layer_x_map.items())
    print(post_ids)
    # print(f"layer map: {layer_x_map}")
    root_id_to_index.update(layer_x_map)
    total_idxs += len(layer_x_map.keys())
    print(total_idxs)



# def update_synapse_idx(neu_array, root_id_to_layer_local):
#     layer_x_map = {
#         rid: (layer_idx, local_idx) for rid, layer_idx, local_idx in root_id_to_layer_local
#     }
#     root_id_to_layer_local.update(layer_x_map)
    

def create_model(conn):
    
    
    global root_id_to_global_index
    global global_idx_to_layer_local
    global previous_idxs
    previous_idxs = 0


    root_id_to_global_index = {}
    global_idx_to_layer_local = {}
    
    num_trials = 5
    sugar_samples = sugar_GRNs['root_id'].sample(num_trials, random_state=1)
    bitter_samples = bitter_GRNs['root_id'].sample(num_trials, random_state=1)

    grn_ids = pd.concat([sugar_GRNs['root_id'], bitter_GRNs['root_id']])
    grn_ids = pd.concat([sugar_samples, bitter_samples])
    print("Printing sugar_GRN_IDs**********************")
    # print(sugar_GRN_IDs)

    # Layers in the 10s: sugar
    # Layers in the 20s: bitter
    running_layer_idx = 0
    
    def from_input_GRNs(grn_ids):
            root_id_to_global_index, global_idx_to_layer_local
            post_ids_by_layer = {}
            post_ids_by_layer[-1] = 0
            layer_offsets = []
            offset = 0
            layer_offsets.append(post_ids_by_layer[-1])
            pre_offset = 0
            post_offset = 0
            previous_idxs = 0
            previous_idxs += len(grn_ids)
            
            original_G = NeuronGroup(len(grn_ids), 
                                eqs, 
                                threshold='v >= v_thresh', 
                                reset='''
                                    v = c
                                    u += d
                                ''', 
                                method='rk4')
            

            # PoissonInput(target=sugar_G, target_var='v', N=len(sugar_G), rate=20*Hz, weight=0.2*mV)
            print("adding GRNs")
            update_index_map(grn_ids, root_id_to_global_index)
            # print(root_id_to_global_index)
            post_ids_by_layer[0] = post_ids
            layer_offsets.append(post_ids)
            for local_idx, rid in enumerate(grn_ids):
                global_idx_to_layer_local[root_id_to_global_index[rid]] = (0, local_idx)
            print(f"root to layer: {global_idx_to_layer_local}")


            
            previous_order_root_ids = grn_ids
            layer_Gs = [original_G]
            synapses_list = []
            all_layers = [original_G]
            neuropils_by_layer = {}
            for l in range(1, 4):
                next_order_root_ids = get_n_order_neurons(previous_order_root_ids, df_conn, 2, True, "sugar", True)
                new_next_order_idxs = [rid for rid in next_order_root_ids if rid not in root_id_to_global_index]
                print(f"new_ids length: {len(new_next_order_idxs)}")
                print(f"type of next order IDs: {type(next_order_root_ids)}")
                print(f"type of previous IDs: {type(previous_order_root_ids)}")
                # print(f"sugar GRNs: {sugar_GRN_IDs}")
                # print(f"next order GRNs: {next_order_root_ids}")
                print(f"adding next order ids for {l} order")
                print(f"length of next order ids: {len(next_order_root_ids)}")

                layer_idx = l # TODO: add running_idx for multiple stimuli

                # print(previous_order_root_ids)
                # print(next_order_root_ids)
                print(len(root_id_to_global_index))  

                next_order_G = NeuronGroup(
                    len(new_next_order_idxs),
                    eqs,
                    threshold='v >= v_thresh', 
                    reset='''
                    v = c
                    u += d
                    ''', 
                    method='euler')
                
                print(global_idx_to_layer_local)
                update_index_map(next_order_root_ids, root_id_to_global_index)
                post_ids_by_layer[l] = post_ids
                layer_offsets.append(post_ids)
                for local_idx, rid in enumerate(new_next_order_idxs):
                    global_idx_to_layer_local[root_id_to_global_index[rid]] = (layer_idx, local_idx)
                print(f"root to layer: {global_idx_to_layer_local}")
                

                # next_order_G = NeuronGroup(len(next_order_root_ids), eqs, threshold='rand() < p', reset='v=-65*mV', method='euler')
                layer_Gs.append(next_order_G)
                print(f"Next order G length (no IDs attached): {len(next_order_G)}")
                print(f"Global post IDs: {post_ids}")

                

                syn = Synapses(layer_Gs[l-1], layer_Gs[l], 
                            model='w : volt',
                            on_pre='v_post += w')
                print(len(layer_Gs[l-1]), len(layer_Gs[l]))
                

                all_layers.append(next_order_root_ids)


                print("l: ", l)
                print(post_ids_by_layer)
                pre_offset += layer_offsets[l-1]  # where source layer starts globally
                post_offset += layer_offsets[l]  # where target layer starts globally
                print(f"length of offsets: {pre_offset}, {post_offset}")

                edges = conn[
                    (conn['pre_root_id'].isin(previous_order_root_ids)) &
                    (conn['post_root_id'].isin(new_next_order_idxs))
                ]
                print(f"Edges: {edges}")
                print(f"Previous order RIDs: {previous_order_root_ids}")
                print(f"Next order RIDs: {next_order_root_ids}")
                print(conn['pre_root_id'].dtype, type((next(iter(previous_order_root_ids)))))
                print(conn['post_root_id'].dtype, type(next(iter(next_order_root_ids))))

                # if l == 1:
                #     for i, row in edges.iterrows():
                #         print(root_id_to_global_index[row['pre_root_id']], root_id_to_global_index[row['post_root_id']])

                # TODO: Add different spiking behavior
                neuron_labels = []
                # for neuron in root_id_to_global_index.values():
                #     tuning_type = classify_neuron_tuning(neuron, root_id_to_global_index, ir94e_GRNs)
                #     print(f'{len(neuron_labels)} visited. {len(root_id_to_global_index.values()) - len(neuron_labels)} remaining in layer {l}')
                #     neuron_labels.append(tuning_type)
                # print(neuron_labels)

                print((edges['post_root_id'].nunique()))
                print(len([idx for rid in edges['post_root_id'] for idx in [root_id_to_global_index[rid]]]))

                # Find ending neuropil
                neuropils = df_conn[df_conn['post_root_id'].isin(edges['post_root_id'])]['neuropil']
                neuropils_by_layer[l] = neuropils

                update_index_map(edges['post_root_id'], root_id_to_global_index)
                print(len(root_id_to_global_index.keys()))

                print(root_id_to_global_index)


                pre_indices_global = [root_id_to_global_index[rid] for rid in edges['pre_root_id']]
                post_indices_global = [root_id_to_global_index[rid] for rid in edges['post_root_id']]
                print("***LENGTHS:", len(pre_indices_global))
                print(len(post_indices_global))

                # Collect synaptic counts for each row in connections.csv
                mask = conn['pre_root_id'].isin(edges['pre_root_id']) & conn['post_root_id'].isin(edges['post_root_id'])

                result = conn[mask]
                syn_count_list = result['syn_count'].tolist()
                print("Syn count list: ", syn_count_list)

                synapses_list.append({
                    'syn': syn,
                    'source_layer': l-1,
                    'target_layer': l,
                    'syn_count_list': syn_count_list
                })
                # print(synapses_list)


                # Map back to local:
                pre_indices_local = []
                post_indices_local = []
                print(zip(edges['pre_root_id'], edges['post_root_id']))
                print(edges['pre_root_id'], edges['post_root_id'])
                for pre_rid, post_rid in zip(edges['pre_root_id'], edges['post_root_id']):
                    print(f"Pre, post IDs: {pre_rid, post_rid}")
                    pre_layer, pre_local = global_idx_to_layer_local[root_id_to_global_index[pre_rid]]
                    post_layer, post_local = global_idx_to_layer_local[root_id_to_global_index[post_rid]]
                    print(f"global idx to layer: {global_idx_to_layer_local[root_id_to_global_index[pre_rid]]}")
                    print(f"Pre, post layers: {pre_layer, post_layer}")
                    if pre_layer == l-1 and post_layer == l:
                        pre_indices_local.append(pre_local)
                        post_indices_local.append(post_local)
                # print(pre_indices_local)
                # print(post_indices_local)

                print(len(pre_indices_local), len(post_indices_local))

                print(f"Presynaptic group size: {post_ids_by_layer[l-1]}")
                print(f"Postsynaptic group size: {post_ids_by_layer[l]}")
                print(f"Min pre index: {min(pre_indices_local)}")
                print(f"Max pre index: {max(pre_indices_local)}")
                print(f"Min post index: {min(post_indices_local)}")
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

                # print(root_id_to_index)
                # print(len(pre_indices_local))
                # print(post_indices_local)
                syn.connect(
                    i=pre_indices_local, 
                    j=post_indices_local)
                
                weights = []
                # print(syn)
                # print(syn.N)

            

                # assert len(weights) == len(syn.w[:])
                # syn.w[:] = weights
                # print(syn.w)
                # print(weights)
                print("done3.75")

                syn.delay = 1*ms # TODO: should this be different based on the neurons

                # for syn in synapses_list:
                    # print(f"Synapse: {syn['syn']}")
                    # print(f"source layer: {syn['source_layer']}")
                    # print(f"target layer: {syn['target_layer']}")

                
                print(type(next_order_root_ids))
                previous_order_root_ids = pd.DataFrame(list(next_order_root_ids), columns=['root_id'])['root_id']
                previous_idxs += len(previous_order_root_ids)
                # pre_offset = post_offset

            print(neuropils_by_layer)
            filtered_conn = conn[
                conn['pre_root_id'].isin(set(root_id_to_global_index.keys())) &
                conn['post_root_id'].isin(set(root_id_to_global_index.keys()))
            ]

            edge_nt_lookup = dict(
                zip(
                    zip(filtered_conn['pre_root_id'], filtered_conn['post_root_id']),
                    filtered_conn['nt_type']
                )
            )
            # print(edge_nt_lookup)

            layer = 1
            all_weights = {}
            for synapse in synapses_list:
                # print(synapse)
                # print(layer)
                syn_count_list = synapse['syn_count_list']
                synapse = synapse['syn']
                weights = []
                visited = 0
                for idx, (pre_id, post_id) in enumerate(zip(synapse.i[:], synapse.j[:])):
                    # print(pre_id, post_id)
                    syn_count = syn_count_list[idx]
                    visited += 1

                    # Map your local indices to global IDs here:
                    pre_root = get_rid_from_global_idx(root_id_to_global_index, get_global_idx_from_layer_local(global_idx_to_layer_local, layer-1, pre_id))
                    post_root = get_rid_from_global_idx(root_id_to_global_index, get_global_idx_from_layer_local(global_idx_to_layer_local, layer, post_id))
                    # print(pre_root, post_root)

                    nt = edge_nt_lookup.get((pre_root, post_root), None)

                    if nt is not None:
                        if nt in ["GLUT", "ACH"]:
                            sign = 1
                        elif nt == "GABA":
                            sign = -1
                            print(f"Inhibitory: {pre_root, post_root}, {pre_id, post_id}")
                            print(f"{visited} inhibitory visited. {len(synapse.i)-visited} remaining.")
                        else:
                            sign = 1
                        
                    
                        # print((root_id_to_global_index[pre_root], root_id_to_global_index[post_root]), "NT:", nt, "Sign:", sign)
                    else:
                        sign = 1
                    if get_global_idx_from_layer_local(global_idx_to_layer_local, layer, post_id) == 289:
                        print(get_global_idx_from_layer_local(global_idx_to_layer_local, layer, post_id), sign, syn_count*w_syn*sign)

                    # print(w_syn)
                    weights.append(syn_count * w_syn * sign)
                synapse.w[:] = weights
                
                print(synapse.w)
                # print(weights)
                all_weights[layer] = weights

                for i, j, w in zip(synapse.i[:], synapse.j[:], synapse.w[:]):
                    if j == 191:
                        print(f"{i} → {j}: {w}")
            
                print(f"Layer {layer-1} → Layer 1 synapses N:", synapse.N)
                print("Example i:", synapse.i[:10])
                print("Example j:", synapse.j[:10])
                print("Example weights:", synapse.w[:10])
                layer += 1

            


            # for i, syn in enumerate(synapses_list):
            #     print(f"Synapses for layer {i}")
            #     print(f"Pre-synaptic indices (i): {list(syn.i)}")
            #     print(f"Post-synaptic indices (j): {list(syn.j)}")

            print(original_G.v.shape)

            for G in layer_Gs:
                G.v = c
                G.u = b * G.v
                G.I = 0 * mV/ms
            
            layer_Gs[0].I = 20 * mV/ms
            layer_Gs[0].I = 20 * mV/ms
            return layer_Gs, synapses_list, all_weights, neuron_labels, neuropils_by_layer

    # layer_Gs_S, synapses_list_S, all_weights_S, neuron_labels_S = from_input_GRNs(sugar_GRNs)
    # layer_Gs_B, synapses_list_B, all_weights_B, neuron_labels_B = from_input_GRNs(bitter_GRNs)

    layer_Gs, synapses_list, all_weights, neuron_labels, neuropils_by_layer = from_input_GRNs(grn_ids)

    # G.I = '10 * mV/ms' # constant input

    # return (layer_Gs_S, layer_Gs_B), (synapses_list_S, synapses_list_B), root_id_to_global_index, global_idx_to_layer_local, (all_weights_S, all_weights_B), (neuron_labels_S, neuron_labels_B)

    return grn_ids, layer_Gs, synapses_list, all_weights, neuron_labels, neuropils_by_layer

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
                    (df_class["class"] == "gustatory") |
                    (df_class["root_id"].isin(twoNs_dfs[taste])) # TODO: is this the problem? they filter by no GRNS of any modality, and no 2Ns of the same modality; issue is likely ir94e
                )
                grn_ids = set(df_class[grn_filter]["root_id"])
                next_ids -= grn_ids

        # Remove already visited; set the current IDs to the ones discovered in this loop, so they can be explored for 3Ns if applicable
        next_ids -= visited
        visited.update(next_ids)
        current_ids = next_ids
    
    return current_ids

def classify_neuron_tuning(global_idx, root_id_to_idx, ir94e_GRNs):
    try:
        rid = get_rid_from_global_idx(root_id_to_idx, global_idx)
    except ValueError:
        return "Not found"

    # Get neurotransmitter type if available
    if rid in df_conn['post_root_id'].values:
        nt_type = df_conn[df_conn['post_root_id'] == rid]['nt_type'].iloc[0]
    else:
        nt_type = "Not there"

    # Get taste class if available
    if rid in df_conn['post_root_id'].values:
        taste = df_class[df_class['root_id'] == rid]['sub_class'].iloc[0]
    else:
        taste = "Not there"

    # Check ir94e status
    ir94e_status = rid in ir94e_GRNs

    # print(nt_type)
    # print(taste)

    # Return classification
    if taste == "sugar/water":
        return taste
    elif ir94e_status:
        return "ir94e"
    elif nt_type:
        return nt_type
    else:
        return "Not found"

def compare_voltage_to_memory_formation(pathways, memory_neuropils, state_monitors):
    voltages = {}
    state_mon = state_monitors[-1]
    # print(f"State mon.v: {state_mon.v}")
    for path_idx in range(len(pathways)):
        pathway = pathways[path_idx]
        last_root_id = get_rid_from_global_idx(root_id_to_global_index, pathway[-1])
        postsynaptic_conns = df_conn[df_conn['post_root_id'] == last_root_id]
        if postsynaptic_conns['neuropil'].isin(memory_neuropils).any():
            # print(state_mon.record)
            voltages[tuple(pathway)] = state_mon.v[global_idx_to_layer_local[pathway[-1]][1]]
    # print(voltages)

    for path, voltage_trace in voltages.items():
        plt.plot(voltage_trace / mvolt, label=str(path), alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.legend()
    return voltages

def run_experiment():
    # (layer_Gs_S, layer_Gs_B), (synapses_list_S, synapses_list_B), root_id_to_global_index, global_idx_to_layer_local, (all_weights_S, all_weights_B), (neuron_labels_S, neuron_labels_B) = create_model(df_conn)
    grn_ids, layer_Gs, synapses_list, all_weights, neuron_labels, neuropils_by_layer = create_model(df_conn)
    
    print("done3")
    # print(root_id_to_index)
    # print(f"All weights: {all_weights}")
    
    synapse_map = create_neuronal_map(synapses_list, root_id_to_global_index, global_idx_to_layer_local)
    # print("Synapse map:", synapse_map)

    start_indices = []
    for id in grn_ids:
        start_indices.append(root_id_to_global_index[id])

    pathways = trace_pathways_multiple_starts(synapse_map, start_indices, root_id_to_global_index)
    # print("Pathways", pathways)

    path_idx = 3
    last_layer_idx = 2
    params_dict = {}
    for neuron in pathways[path_idx]:
        a = 0.02/ms # time scale of recovery variable u
        b = 0.2/second # sensitivity of u to v
        c = -65*mV # after-spike reset of v
        d = 2*mV/ms # increment of u after spike

    # print(root_id_to_global_index)

    state_monitors_S = []
    # for layer_Gs in (layer_Gs_S, layer_Gs_B):
    state_monitors = []
    for g in layer_Gs:
        M = StateMonitor(g, 'v', record=True)
        state_monitors.append(M)
    
    # print(state_monitors[-1])


    spike_monitors = []
    for g in layer_Gs:
        sm = SpikeMonitor(g)
        spike_monitors.append(sm)
    state_monitors_S = state_monitors

    memory_neuropils = [
        'MB_CA_L', 'MB_CA_R',
        'MB_PED_L', 'MB_PED_R',
        'MB_VL_L', 'MB_VL_R',
        'MB_ML_L', 'MB_ML_R',
        'CRE_L', 'CRE_R',
        'SMP_L', 'SMP_R',
        'SIP_L', 'SIP_R',
        'SLP_L', 'SLP_R',
        'LH_L', 'LH_R'
    ]
    
    
    
    # for i in range(len(synapses_list)):
    #     print("Layer 1 --> Layer 2: ", synapses_list[i].i)
    #     print("Weights: ", synapses_list[i].w)

    # Testing
    # print(root_id_to_layer_and_local_idx)
    layer_idx = 1

    # rid = get_rid_from_global_idx(root_id_to_global_index, 84)
    # print(f"type: {type(rid)}")
    # print(root_id_to_global_index[rid])
    # local_idx = global_idx_to_layer_local[84][1]

    # print(root_id_to_global_index[rid])
    # layer_Gs[layer_idx].I[local_idx] = 500 * mV/ms
    # print(layer_Gs[layer_idx].I[local_idx])
    
    net = Network(layer_Gs + 
                  [s['syn'] for s in synapses_list] + 
                  spike_monitors + 
                  state_monitors)
    print("done3.85")
    net.run(100*ms)
    print("done4")
    # Model synaptic dynamics
    # Implement plasticity rules
    # Specify delays
    # Create a Network object


    spike_data = []
    for l_idx, monitor in enumerate(spike_monitors):
        print(f"Layer {l_idx} spikes:", monitor.count)
        for neuron_idx, t in zip(monitor.i, monitor.t):
            global_idx = get_global_idx_from_layer_local(global_idx_to_layer_local, l_idx, int(neuron_idx))
            spike_data.append((t/ms, global_idx, l_idx, int(neuron_idx)))

    spike_data.sort()

    for path_id, path in enumerate(pathways[:2]):  # First 1
        print(f"Path {path_id}")
        for t, global_id, layer, local_id in spike_data:
            if global_id in path:
                print(f"Global neuron {global_id} in layer {layer} spiked at {t} ms")


    voltages = compare_voltage_to_memory_formation(pathways, memory_neuropils, state_monitors)

    # print(neuron_labels)

    # Plot results
    figure(figsize=(12, 12))

    # monitors0to2 = [state_monitors[0], state_monitors[1], state_monitors[2]]

    subplot(2,1,1)
    # for key in list(pathways)[:3]:
    layer_idx = 0

    path_idx = 4
    def plot_voltages(pathways):
        # for i in range(len(pathways)):
        #     if pathways[i] == [0, 84, 780]:
        #         path_idx = i
        #         print(path_idx)

        # path_idxs = filter_by_sign(pathways, "Positive", 5, all_weights)
        for path_idx in range(18):
            plt.figure(figsize=(8, 4))
            for neuron_idx in pathways[path_idx]:
                print(pathways[path_idx])
                print(neuron_idx)
                print(state_monitors)

                (layer_idx, local_idx) = global_idx_to_layer_local[neuron_idx]
                M = state_monitors[layer_idx]
                rid = get_rid_from_global_idx(root_id_to_global_index, neuron_idx)
                
                if local_idx < len(M.v):
                    plt.plot(
                        M.t/ms, 
                        M.v[local_idx]/mV, 
                        label=f'Path {path_idx} Neuron {neuron_idx} (RID: {rid})',
                        alpha=0.7,
                        linewidth=1.0)
                    print(neuron_idx, M.t, M.v[local_idx])
                
            # layer_idx += 1
            plt.xlabel('Time (ms)')
            plt.ylabel('v (mV)')
            plt.title(f'Membrane potential -- Pathway {path_idx}')
            plt.legend()
            plt.tight_layout()
        

        # TODO: Fix this

        # last_idx = len(pathways[path_idx])-1
        # last_v = state_monitors[last_layer_idx].v
        # last_spike_times = spike_monitors[0].t[spike_monitors[0].i == last_idx]

        # window_ms = 10
        # upstream_spike_times = [spike_monitors[0].t[spike_monitors[0].i == 1] for i in range(len(pathways[path_idx]))]
        # for spike_time in last_spike_times:
        #     print(f"Last neuron spike at {spike_time/ms:.1f} ms")
        #     for i in upstream_spike_times:
        #         count = np.sum(spike_monitors.i == i)
        #         duration = state_monitors.t[-1]/second
        #         rate = count/duration
        #         print(f"Neuron {i} spiked {count} times, rate = {rate:.2f} Hz")


        # subplot(2,1,2)
        # plot(sugar_M.t/ms, sugar_M.u[0]/(mV/ms))
        # xlabel('Time (ms)')
        # ylabel('u (mV/ms)')
        # title('Recovery variable u')
        # print("done5")
        plt.legend()

    def plot_activation_sensitivity(state_mons, spike_mons, paths):
        
        plt.figure()
        for path in paths[:3]:
            neuron_2 = global_idx_to_layer_local[path[2]][1]
            v2 = state_mons[2].v[neuron_2]
            plt.plot(state_mons[2].t/ms, v2/mV, label=f'N2: {path[2]}')
        plt.legend()

        plt.figure(figsize=(10, 6))
        slopes = []
        for path in [path for path in paths if len(path) == 4]:
            idx0, idx1, idx2, idx3 = path
            net_v = state_mons[0].v[global_idx_to_layer_local[idx0][1]] + state_mons[1].v[global_idx_to_layer_local[idx1][1]] + state_mons[2].v[global_idx_to_layer_local[idx2][1]] + state_mons[3].v[global_idx_to_layer_local[idx3][1]]
            # plt.plot(state_mons[0].t / ms, net_v, label=f"Path {idx0}-{idx1}-{idx2}")
            slope, _ = np.polyfit(state_mons[0].t / ms, net_v, 1)
            slopes.append(slope)
        plt.bar(range(len(paths)), slopes)
        plt.xlabel("Path Index")
        plt.ylabel("Slope of Activation Sensitivity")
        plt.title("Sensitivity Slope per Path")
        plt.legend()
        
        
        plt.figure(figsize=(10, 6))
        net_voltage_upstream = []
        net_voltage_upstream_list = []
        times = state_mons[0].t
        for i, path in enumerate(paths):
            net_v = 0
            # net_voltage_upstream = []
            for t in range(len(state_mons[0].v)):
                for layer, idx in enumerate(path[:3]):
                    mon_layer = layer
                    neuron_idx = global_idx_to_layer_local[idx][1]
                    # print(mon_layer, neuron_idx)
                    # print(spike_mons[layer].i[neuron_idx], spike_mons[layer].t)
                    v = state_mons[mon_layer].v[neuron_idx][t]
                    net_v += v
                    print(idx, v, net_v)
                    if t == 0:
                        print(f"Neuron {idx} in monitor {mon_layer} has v[0] = {v}")
                # net_v = sum(state_mons[layer].v[global_idx_to_layer_local[idx][1]][t] for layer, idx in enumerate(path[:2]))

                net_voltage_upstream.append(net_v)
                
            net_voltage_upstream_list.append(net_voltage_upstream)
            
        for i, net_v in enumerate(net_voltage_upstream_list):  # list of lists of net voltages
            plt.plot(times, net_v, label=f'Path {i}')    
            # X = np.array(times).reshape(-1, 1)
            # y = np.array(net_voltage_upstream)

            # model = LinearRegression().fit(X, y)

            # y_pred = model.predict(X)
            # plt.plot(X, y_pred, '--', label=f'Pathway {i+1}')

        # plt.plot(X, y, label='Net voltage')
        plt.title(f'Activation Sensitivity: Net Voltage Over Time per Pathway')
        plt.xlabel('Time (ms')
        plt.ylabel('Net voltage (mV)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    def plot_neuropils_percentages(neuropil_dict_by_layer):
        final_layer = max(neuropil_dict_by_layer.keys())
        final_neuropils = neuropil_dict_by_layer[final_layer]

        counts = Counter(final_neuropils)

        neuropils = list(counts.keys())
        sizes = list(counts.values())
        print(counts)

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, 
                labels=neuropils,
                autopct='%1.1f%%',
                startangle=140,
                labeldistance=1.1,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
                )
        plt.axis('equal')
        plt.tight_layout()

        
        

    def animate_spiking(pathways, state_monitors, spike_data, graph_type):
        pathway_neurons = set(pathways[path_idx])
        spike_times = [(t, gid) for t, gid, _, _ in spike_data if gid in pathway_neurons]
        print(spike_times)

        times = [t for t, _ in spike_times]
        if times:
            max_time = max(times)
        else:
            max_time = 1000

        if graph_type == "dots":
            # Spike dots animation
            fig, ax = plt.subplots()

            points = {}
            colors = {}

            max_time = 0
            spike_times = []


            

            for i, gid in enumerate(pathway_neurons):
                color = plt.cm.viridis(i / len(pathway_neurons))
                colors[gid] = color
                point, = ax.plot([], [], 'o', color=color, label=f'Neuron {gid}')
                points[gid] = point

            # line, = ax.plot([], [], lw=2)
            ax.set_xlim(0, max_time)
            ax.set_ylim(min(gid for _, gid in spike_times) - 1,
                    max(gid for _, gid in spike_times) + 1)
            # ax.set_title("Membrane Potential Over Time")

            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Neuron ID")
            ax.set_title("Spiking Activity Over Time")

            scat = ax.scatter([], [], s=30, color='crimson')

            # def init():
            #     line.set_data([], [])
            #     return (line,)
            x_vals = []
            y_vals = []

            def update(frame_time):
                # Add any spikes that occur at this time
                new_spikes = [(t, gid) for (t, gid) in spike_times if abs(t - frame_time) < 0.25]
                for t, gid in new_spikes:
                    x_vals.append(t)
                    y_vals.append(gid)
                scat.set_offsets(np.column_stack((x_vals, y_vals)))
                return scat,

            

            ani = FuncAnimation(fig, update, frames=arange(0, max_time, 1), interval=50, blit=True)
            plt.show()
        elif graph_type == "voltage":
            # Spike voltage animation (all neurons in the pathway)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Membrane potential (mV)")
            ax.set_title("Membrane Potentials of Pathway Neurons")

            # Pick colors for each neuron
            colors = plt.cm.viridis(np.linspace(0, 1, len(pathway_neurons)))

            # Collect time and voltage data for each neuron
            time_array = None
            voltages = []
            for neuron_idx in pathway_neurons:
                layer_idx, local_idx = global_idx_to_layer_local[neuron_idx]
                monitor = state_monitors[layer_idx]

                if time_array is None:
                    time_array = monitor.t / ms  # shared time array

                if local_idx < len(monitor.v):
                    voltages.append(monitor.v[local_idx] / mV)
                else:
                    voltages.append(np.full_like(monitor.t / ms, np.nan))  # skip if invalid

            # Create an empty line for each neuron
            lines = []
            for i, neuron_idx in enumerate(pathway_neurons):
                (line,) = ax.plot([], [], label=f"Neuron {neuron_idx}", color=colors[i], linewidth=1.5)
                lines.append(line)

            ax.legend(loc='upper right')
            ax.set_xlim(0, time_array[-1])
            ax.set_ylim(min(np.nanmin(v) for v in voltages) - 5, max(np.nanmax(v) for v in voltages) + 5)

            # Animation update function
            def update(frame_idx):
                t = time_array[:frame_idx]
                for i, line in enumerate(lines):
                    line.set_data(t, voltages[i][:frame_idx])
                return lines

            print(len(time_array))
            # Animate over all time steps
            ani = FuncAnimation(
                fig,
                update,
                frames=range(0, int(max_time), 4),
                interval=5,  # ms between frames (adjust for speed)
                blit=True
            )

            print("exporting")
            ani.save('voltage_animation.gif', writer='pillow', fps=30)
            print("exported")

            
        
    def node_diagram(pathways, all_weights):
        nodes = []
        for path in pathways:
            for neuron in path:
                if neuron not in nodes:
                    nodes.append(neuron)
        print(f"Nodes: {nodes}")
        edges = []
        for i in list(synapse_map.keys())[:10]:
            for j in synapse_map[i]:
                edges.append((i, j, all_weights[i]))
            

    # node_diagram(pathways, all_weights)


    print(pathways)
    print(root_id_to_global_index)
    animate_spiking(pathways, state_monitors, spike_data, "voltage")
    plot_voltages(pathways)
    plot_neuropils_percentages(neuropils_by_layer)
    # plot_activation_sensitivity(state_monitors, spike_monitors, [pathways[i] for i in range(5)])
    plt.show(block=True)


# TODO: fix the sign issue
def filter_by_sign(pathways, sign, num_paths, all_weights):
    path_idxs = []
    
    for idx, pathway in enumerate(pathways):
        if len(path_idxs) > 4:
            break
        pathway = pathways[idx]
        signs = []
        # print(all_weights)
        for layer in all_weights:
            weights = all_weights[layer]
            node_local_idx = global_idx_to_layer_local[pathway[layer]][1]
            # print(pathway[layer], weights[node_local_idx])
            for i in enumerate(weights[0]):
                print(i, weights[0][i])
            if weights[node_local_idx]/mV < 0:
                print("negative")
                signs.append("Negative")
            elif weights[node_local_idx]/mV > 0:
                print("positive")
                signs.append("Positive")
            print(pathway[layer], layer, node_local_idx, weights[node_local_idx])
            print()
        print(signs)
        if all(item == sign for item in signs):
            path_idxs.append(idx)
    print(path_idxs)
    return path_idxs



def create_neuronal_map(synapses_list, root_id_to_global_index, root_id_to_layer_local):
    print("starting creation")
    # print(root_id_to_global_index)
    synapse_map = {}
    for synapse in synapses_list:
        # print(synapse.i, synapse.j)
        explored = 0

        for pre_idx, post_idx in zip(synapse['syn'].i, synapse['syn'].j):
            # print(synapse['source_layer'], synapse['target_layer'])

            global_pre_idx = get_global_idx_from_layer_local(root_id_to_layer_local, synapse['source_layer'], int(pre_idx))
            global_post_idx = get_global_idx_from_layer_local(root_id_to_layer_local, synapse['target_layer'], int(post_idx))
            # if synapse['target_layer'] == 1:
            #     print(f"Pre idx: {global_pre_idx}\nBefore: {pre_idx} {synapse['source_layer']} {root_id_to_layer_local[global_pre_idx]}")
            #     print(f"Post idx: {global_post_idx}\nBefore: {post_idx}")
            synapse_map[int(global_pre_idx)] = synapse_map.get(int(global_pre_idx), []) + [int(global_post_idx)]
            # if synapse['target_layer'] == 1:
            #     print(global_pre_idx, synapse_map[global_pre_idx])
            explored += 1
            print(f"{explored} explored. {len(synapse['syn'].i)-explored} remaining.")
        
    # print(synapse_map)
    for num in synapse_map:
        continue
        # print(num)
        # print(synapse_map[num][0])
        # print(f"{get_rid_from_idx(root_id_to_index, num)} -> {get_rid_from_idx(root_id_to_index, synapse_map[num][0])}") # get_rid_from_idx works well
        # print(root_id_to_index[get_rid_from_idx(root_id_to_index, num)])
    print("returning")
    return synapse_map

def trace_pathways(synapse_map, start_indices, rid_dict, max_depth=5, max_paths=1000):
    print("starting tracing")
    pathways = []
    path_idx = 0
    for start_idx in start_indices:
        stack = [[start_idx]]
        collected_paths = []
        visited_paths = set()
        print("created variables")
        # print("Synapse map[0]: ", synapse_map[0])
        # print("Synapse map[57]: ", synapse_map[57])

        idx_instances = 0
        last_used_idx = 0
        while stack:
            # print(f"\n\n\n\n\n****STACK:***\n\n {stack}")
            current_path = stack.pop()
            while current_path[0] == last_used_idx and idx_instances > 2:
                current_path = stack.pop()
                print(current_path)
                print(idx_instances)
            if current_path[0] != last_used_idx:
                idx_instances = 0
            
            nodes = []
            for node in current_path:
                nodes.append(get_rid_from_global_idx(rid_dict, node))
            last_node = current_path[-1]
            first_node = current_path[0]

            # print("new path: ", current_path, nodes)
            
            if len(current_path) >= max_depth:
                pathways[path_idx] = current_path
                path_idx += 1
                continue
            next_nodes = synapse_map.get(last_node, [])
            last_used_idx = current_path[0]
            idx_instances += 1
            # print(f"Next nodes: {next_nodes}")
            # print(pathways)
            
            if not next_nodes:
                # print("not next nodes")
                pathways.append(current_path)
                # print(path_idx, pathways[path_idx])
                path_idx += 1
            else:
                # print("is next nodes")
                for n in next_nodes:
                    if n in current_path:
                        continue
                    stack.append(current_path + [n])
                    # print(stack[-1])
            
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

def trace_pathways_multiple_starts(synapse_map, start_indices, root_id_to_global_index, max_pathways=3):
    all_paths = []
    for start_idx in start_indices:
        stack = [[start_idx]]
        collected_paths = []
        visited_paths = set()

        while stack and len(collected_paths) < max_pathways:
            current_path = stack.pop()
            last_node = current_path[-1]

            # Avoid duplicates
            path_tuple = tuple(current_path)
            if path_tuple in visited_paths:
                continue
            visited_paths.add(path_tuple)

            # Get downstream nodes from synapse_map
            downstream_nodes = synapse_map.get(last_node, [])

            if not downstream_nodes:
                # End of path - collect it
                collected_paths.append(current_path)
                continue

            for next_node in downstream_nodes:
                if next_node not in current_path:  # Avoid cycles
                    stack.append(current_path + [next_node])

        all_paths.extend(collected_paths)

    return all_paths



    
run_experiment()   

plt.show()
        



