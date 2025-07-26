from brian2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

start_scope()

df_conn = pd.read_csv("database/connections.csv")
df_class = pd.read_csv("database/classification.csv")
sugar_GRNs = pd.read_csv("database/sugar_GRNs.csv")
bitter_GRNs = pd.read_csv("database/bitter_GRNs.csv")
lowsalt_GRNs = pd.read_csv("database/lowsalt_GRNs.csv")
water_GRNs = pd.read_csv("database/water_GRNs.csv")
sugar_2Ns = pd.read_csv("output_csvs/sugar_2Ns.csv")
bitter_2Ns = pd.read_csv("output_csvs/bitter_2Ns.csv")
lowsalt_2Ns = pd.read_csv("output_csvs/lowsalt_2Ns.csv")
water_2Ns = pd.read_csv("output_csvs/water_2Ns.csv")

total_idxs = 0
post_ids = 0
previous_idxs = 0
pre_offset = 0

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
    previous_idxs = 0

    

    root_id_to_global_index = {}
    global_idx_to_layer_local = {}
    post_ids_by_layer = {}
    post_ids_by_layer[-1] = 0
    layer_offsets = []
    offset = 0
    layer_offsets.append(post_ids_by_layer[-1])
    pre_offset = 0
    post_offset = 0
    

    grn_ids = sugar_GRNs['root_id']
    print("Printing sugar_GRN_IDs**********************")
    # print(sugar_GRN_IDs)

    original_G = NeuronGroup(len(grn_ids), 
                          eqs, 
                          threshold='v >= v_thresh', 
                          reset='''
                            v = c
                            u += d
                          ''', 
                          method='rk4')
    # PoissonInput(target=sugar_G, target_var='v', N=len(sugar_G), rate=20*Hz, weight=0.2*mV)
    print("adding sugar GRNs")
    update_index_map(grn_ids, root_id_to_global_index)
    # print(root_id_to_global_index)
    previous_idxs += len(grn_ids)
    post_ids_by_layer[0] = post_ids
    layer_offsets.append(post_ids)
    for local_idx, rid in enumerate(grn_ids):
        global_idx_to_layer_local[root_id_to_global_index[rid]] = (0, local_idx)
    print(f"root to layer: {global_idx_to_layer_local}")


    
    previous_order_root_ids = grn_ids
    layer_Gs = [original_G]
    synapses_list = []
    all_layers = [original_G]
    for l in range(1, 3):
        next_order_root_ids = get_n_order_neurons(previous_order_root_ids, df_conn, 2, True, "sugar", True)
        new_next_order_idxs = [rid for rid in next_order_root_ids if rid not in root_id_to_global_index]
        print(f"new_ids length: {len(new_next_order_idxs)}")
        print(f"type of next order IDs: {type(next_order_root_ids)}")
        print(f"type of previous IDs: {type(previous_order_root_ids)}")
        # print(f"sugar GRNs: {sugar_GRN_IDs}")
        # print(f"next order GRNs: {next_order_root_ids}")
        print(f"adding next order ids for {l} order")
        print(f"length of next order ids: {len(next_order_root_ids)}")

        layer_idx = l

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
        # if l == 1:
        #     for i, row in edges.iterrows():
        #         print(root_id_to_global_index[row['pre_root_id']], root_id_to_global_index[row['post_root_id']])

        print((edges['post_root_id'].nunique()))
        print(len([idx for rid in edges['post_root_id'] for idx in [root_id_to_global_index[rid]]]))

        update_index_map(edges['post_root_id'], root_id_to_global_index)
        print(len(root_id_to_global_index.keys()))


        missing_pres = [rid for rid in edges['pre_root_id'] if rid not in root_id_to_global_index]
        missing_posts = [rid for rid in edges['post_root_id'] if rid not in root_id_to_global_index]

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
        print(synapses_list)

        # Map back to local:
        pre_indices_local = []
        post_indices_local = []
        for pre_rid, post_rid in zip(edges['pre_root_id'], edges['post_root_id']):
            pre_layer, pre_local = global_idx_to_layer_local[root_id_to_global_index[pre_rid]]
            post_layer, post_local = global_idx_to_layer_local[root_id_to_global_index[post_rid]]
            
            if pre_layer == l-1 and post_layer == l:
                pre_indices_local.append(pre_local)
                post_indices_local.append(post_local)
        print(pre_indices_local)
        print(post_indices_local)

        print(len(pre_indices_local), len(post_indices_local))

        print(f"Presynaptic group size: {post_ids_by_layer[l-1]}")
        print(f"Postsynaptic group size: {post_ids_by_layer[l]}")
        # print(f"Max pre index: {pre_indices.max()}")
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
        print(syn)
        print(syn.N)

    

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

        

        previous_order_root_ids = pd.DataFrame(list(next_order_root_ids), columns=['root_id'])['root_id']
        previous_idxs += len(previous_order_root_ids)
        # pre_offset = post_offset

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
    for synapse in synapses_list:
        print(synapse)
        print(layer)
        syn_count_list = synapse['syn_count_list']
        synapse = synapse['syn']
        weights = []
        for idx, (pre_id, post_id) in enumerate(zip(synapse.i[:], synapse.j[:])):
            print(pre_id, post_id)
            syn_count = syn_count_list[idx]

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
                    print(f"Inhibitory: {pre_root, post_root}")
                else:
                    sign = 1
                
            
                print((root_id_to_global_index[pre_root], root_id_to_global_index[post_root]), "NT:", nt, "Sign:", sign)
            else:
                sign = 1

            print(w_syn)
            weights.append(syn_count * w_syn * sign)
        synapse.w[:] = weights
        
        print(synapse.w)
        print(weights)
    
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
        G.I = 10 * mV/ms
    
    layer_Gs[0].I[0] = 200 * mV/ms

    # G.I = '10 * mV/ms' # constant input

    # Monitors
    spikemon = SpikeMonitor(original_G)
    sugar_M = StateMonitor(original_G, 'v', record=True)

    return layer_Gs, synapses_list, spikemon, sugar_M, root_id_to_global_index, global_idx_to_layer_local

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

def run_experiment():
    layer_Gs, synapses_list, spikemon, sugar_M, root_id_to_global_index, global_idx_to_layer_local = create_model(df_conn)
    print("done3")
    # print(root_id_to_index)
    
    synapse_map = create_neuronal_map(synapses_list, root_id_to_global_index, global_idx_to_layer_local)
    # print("Synapse map:", synapse_map)
    pathways = trace_pathways(synapse_map, 0, root_id_to_global_index)
    # print("Pathways", pathways)

    # print(root_id_to_global_index)

    state_monitors = []
    for g in layer_Gs:
        M = StateMonitor(g, 'v', record=True)
        state_monitors.append(M)

    spike_monitors = []
    for g in layer_Gs:
        sm = SpikeMonitor(g)
        spike_monitors.append(sm)
    
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

    path_idx = 4
    # for i in range(len(pathways)):
    #     if pathways[i] == [0, 84, 780]:
    #         path_idx = i
    #         print(path_idx)
    for path_idx in range(5):
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
            
        # layer_idx += 1
        plt.xlabel('Time (ms)')
        plt.ylabel('v (mV)')
        plt.title(f'Membrane potential -- Pathway {path_idx}')
        plt.legend()
        plt.tight_layout()

    # subplot(2,1,2)
    # plot(sugar_M.t/ms, sugar_M.u[0]/(mV/ms))
    # xlabel('Time (ms)')
    # ylabel('u (mV/ms)')
    # title('Recovery variable u')
    # print("done5")
    plt.legend()

def create_neuronal_map(synapses_list, root_id_to_global_index, root_id_to_layer_local):
    print("starting creation")
    print(root_id_to_global_index)
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

def trace_pathways(synapse_map, start_idx, rid_dict, max_depth=5, max_paths=1000):
    print("starting tracing")
    pathways = []
    path_idx = 0
    stack = [[start_idx]]
    print("created variables")
    # print("Synapse map[0]: ", synapse_map[0])
    # print("Synapse map[57]: ", synapse_map[57])

    while stack:
        # print(f"\n\n\n\n\n****STACK:***\n\n {stack}")
        current_path = stack.pop()
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


    
run_experiment()   

plt.show()
        



