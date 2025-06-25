import pandas as pd
import numpy as np
# import caveclient
import seaborn as sns

# client = caveclient.CAVEclient()
# client.auth.setup_token(make_new=True)


# Variables
classifications = pd.read_csv("classification.csv")
connections = pd.read_csv("connections.csv")
labels = pd.read_csv("labels.csv")

print(classifications.columns)
print(connections.columns)

pd.set_option("display.max_colwidth", None)

# print(classifications["class"].unique())
# print(classifications["flow"].unique())
# print(connections["neuropil"].unique())
# print(classifications["sub_class"].unique())


# Function definitions
def find_total_upstream_neurons(neuron_df, connections):
    visited = set(neuron_df["post_root_id"])
    frontier = set(neuron_df["post_root_id"])
    while len(frontier) > 0:
        # print(f"Frontier size: {len(frontier)}\nVisited: {len(visited)}")
        new_neurons = set()
        new_connections = connections[connections["post_root_id"].isin(frontier)]
        upstream_neurons = set(new_connections["pre_root_id"])
        if new_neurons == 0:
            break  
        for n in upstream_neurons:
            if n not in visited:
                new_neurons.add(n)
        visited.update(new_neurons)
        frontier = new_neurons
    return list(visited)

def find_total_downstream_neurons(neuron_df, connections):
    visited = set(neuron_df["pre_root_id"])
    frontier = set(neuron_df["pre_root_id"])
    while len(frontier) > 0:
        # print(f"Frontier size: {len(frontier)}\nVisited: {len(visited)}")
        new_neurons = set()
        new_connections = connections[connections["pre_root_id"].isin(frontier)]
        downstream_neurons = set(new_connections["post_root_id"])
        if new_neurons == 0:
            break  
        for n in downstream_neurons:
            if n not in visited:
                new_neurons.add(n)
        visited.update(new_neurons)
        frontier = new_neurons
    return list(visited)

def find_print_total_neurons(neuron_list, characteristic, merge):
    if merge:
        combined_neuron_list = get_connections(neuron_list)
    else:
        combined_neuron_list = neuron_list
    neurons_count = len(combined_neuron_list["pre_root_id"].unique())
    upstream = find_total_upstream_neurons(combined_neuron_list, connections)
    downstream = find_total_downstream_neurons(combined_neuron_list, connections)
    combined_unique_neurons = get_total_neurons(combined_neuron_list)
    
    print(f"\n\nNumber of {characteristic} neurons = {neurons_count}")
    print("upstream neurons: ", len(upstream) - neurons_count)
    print("downstream neurons: ", len(downstream) - neurons_count)
    print(f"Total {characteristic}-associated neurons: {len(combined_unique_neurons)}")
    

def get_connections(neuron_list):
    return pd.merge(neuron_list, connections, left_on='root_id', right_on='pre_root_id', how='inner')

def get_total_neurons(combined_neuron_list):
    upstream = find_total_upstream_neurons(combined_neuron_list, connections)
    downstream = find_total_downstream_neurons(combined_neuron_list, connections)
    combined_unique_neurons = list(set(upstream) | set(downstream))
    return combined_unique_neurons

def find_shared_nodes(neuron_list1, neuron_list2, merge1, merge2):
    if merge1:
        combined_neuron1_list = get_connections(neuron_list1)
    else:
        combined_neuron1_list = neuron_list1
    if merge2:
        combined_neuron2_list = get_connections(neuron_list2)
    else:
        combined_neuron2_list = neuron_list2
    total_neurons1 = get_total_neurons(combined_neuron1_list)
    total_neurons2 = get_total_neurons(combined_neuron2_list)
    shared_neurons = list(set(total_neurons1) & set(total_neurons2))
    return shared_neurons
    
def get_n_order_neurons(classifications_neu_list, connections, order):
    current_set = set(classifications_neu_list)
    visited = set(classifications_neu_list)

    for i in range(order-1):
        pre_neu_list = connections[connections['pre_root_id'].isin(current_set)]
        post_neu_list = set(connections[pre_neu_list['post_root_id']])
        if len(post_neu_list) > 0:
            for neu in post_neu_list:
                visited += neu
        current_set = post_neu_list
    return visited


# Gr5a (Sweet sensing)
sweet_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "sugar/water")]
find_print_total_neurons(sweet_GRN_list, "sweet", True)
print(get_n_order_neurons(sweet_GRN_list, connections, 2))

bitter_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "bitter")]
find_print_total_neurons(bitter_GRN_list, "bitter", True)

# Ir76b (Protein sensing)
salty_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "low-salt")]
find_print_total_neurons(salty_GRN_list, "salty", True)


# Serotonergic neurons
serotonergic_neurons = connections[connections["nt_type"] == "SER"]
find_print_total_neurons(serotonergic_neurons, "serotonine", False)


# Dopaminergic neurons
dopaminergic_neurons = connections[connections["nt_type"] == "DA"]
find_print_total_neurons(dopaminergic_neurons, "dopamine", False)


# Ir94e neurons
ir94e_neurons = labels[labels["label"].str.contains("Ir94e")]
print(f"\n\nTotal shared nodes between Gr5a and Ir94e (including upstream/downstream nodes): {len(find_shared_nodes(sweet_GRN_list, ir94e_neurons, True, True))}")
print(f"Shared neurons between Gr5a and Ir94e: {len(list(set(sweet_GRN_list) & set(ir94e_neurons)))}")