import pandas as pd
import numpy as np
# import caveclient
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# client = caveclient.CAVEclient()
# client.auth.setup_token(make_new=True)


# Variables
classifications = pd.read_csv("database\\classification_630.csv") # change this back sometime
connections = pd.read_csv("database\\connections_630.csv")
connections = connections[connections["syn_count"] >= 5]
labels = pd.read_csv("database\\labels.csv")

sugar_GRNs = pd.read_csv("database\\sugar_GRNs.csv")
water_GRNs = pd.read_csv("database\\water_GRNs.csv")
bitter_GRNs = pd.read_csv("database\\bitter_GRNs.csv")
lowsalt_GRNs = pd.read_csv("database\\lowsalt_GRNs.csv")


three_n_threshold = 50

# print(classifications.columns)
# print(connections.columns)

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

def find_shared_nodes(neuron_list1, neuron_list2, merge1, merge2, order):
    if merge1:
        combined_neuron1_list = get_connections(neuron_list1)
    else:
        combined_neuron1_list = neuron_list1
    if merge2:
        combined_neuron2_list = get_connections(neuron_list2)
    else:
        combined_neuron2_list = neuron_list2
    n_order_neus1 = get_n_order_neurons(combined_neuron1_list, connections, order)
    n_order_neus2 = get_n_order_neurons(combined_neuron2_list, connections, order)
    shared_neurons = list(set(n_order_neus1) & set(n_order_neus2))
    return shared_neurons

    
def get_n_order_neurons(neu_list, connections, order, merge):
    # print(neu_list)
    if merge:
        neu_list = get_connections(neu_list)
    # print(neu_list)
    current_ids = set(neu_list['root_id'])
    visited = set(neu_list['root_id'])

    if order == 3:
        connections = connections[connections['syn_count'] >= 5]

    for i in range(order-1):
        pre_neu_list = connections[connections['pre_root_id'].isin(current_ids)]
        # print("len of pre_neu_list:", len(pre_neu_list))
        post_neu_list = set(pre_neu_list['post_root_id'])
        # print("len of post_neu_list:", len(post_neu_list))
        
        post_neu_list -= visited
        visited.update(post_neu_list)
        current_ids = post_neu_list
        # print(len(visited))
        # print(len(current_ids))
    return current_ids


def total_shared_order_neurons(sugar, water, bitter, salt, connections, order):
    second_order = {
        "sweet": get_n_order_neurons(sugar, connections, order, merge=True),
        "water": get_n_order_neurons(water, connections, order, merge=True),
        "bitter": get_n_order_neurons(bitter, connections, order, merge=True),
        "ir94e": get_n_order_neurons(salt, connections, order, merge=True)
    }

    # print(second_order["sweet"], len(second_order["sweet"]))
    # print(get_n_order_neurons(sugar, connections, order, True))
    for r in range(2, 5):  # combinations of size 2 to 4
        for combo in combinations(second_order.keys(), r):
            # print(set(second_order[combo[0]]) & set(second_order[combo[1]]))
            shared_neurons = set.intersection(*[second_order[name] for name in combo])
            print(f"Total shared neurons in {list(combo)}: {len(shared_neurons)}")

total_shared_order_neurons(sugar_GRNs, water_GRNs, bitter_GRNs, lowsalt_GRNs, connections, 2)

# Gr5a (Sweet sensing)
sweet_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "sugar/water")]
find_print_total_neurons(sweet_GRN_list, "sweet", True)
print(get_n_order_neurons(sweet_GRN_list, connections, 2))

bitter_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "bitter")]
find_print_total_neurons(bitter_GRNs, "bitter", True)
print(f"2nd order neurons: {len(get_n_order_neurons(bitter_GRNs, connections, 2, True))}")
print(f"3rd order neurons: {len(get_n_order_neurons(bitter_GRNs, connections, 3, True))}")


# Ir76b (Protein sensing)
salty_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "low-salt")]
find_print_total_neurons(lowsalt_GRNs, "salty", True)
print(f"2nd order neurons: {len(get_n_order_neurons(lowsalt_GRNs, connections, 2, True))}")
print(f"3rd order neurons: {len(get_n_order_neurons(lowsalt_GRNs, connections, 3, True))}")


find_print_total_neurons(water_GRNs, "water", True)
print(f"2nd order neurons: {len(get_n_order_neurons(water_GRNs, connections, 2, True))}")
print(f"3rd order neurons: {len(get_n_order_neurons(water_GRNs, connections, 3, True))}")



# # Serotonergic neurons
# serotonergic_neurons = connections[connections["nt_type"] == "SER"]
# find_print_total_neurons(serotonergic_neurons, "serotonine", False)
# print(f"2nd order neurons: {len(get_n_order_neurons(serotonergic_neurons, connections, 2, False))}")

# # Dopaminergic neurons
# dopaminergic_neurons = connections[connections["nt_type"] == "DA"]
# find_print_total_neurons(dopaminergic_neurons, "dopamine", False)
# print(f"2nd order neurons: {len(get_n_order_neurons(dopaminergic_neurons, connections, 2, False))}")


# Ir94e neurons
ir94e_neurons = labels[labels["label"].str.contains("Ir94e")]
print(f"\n\nTotal shared nodes between Gr5a and Ir94e (including upstream/downstream nodes): {len(find_shared_nodes(sweet_GRN_list, ir94e_neurons, True, True))}")
print(f"Shared neurons between Gr5a and Ir94e: {len(list(set(sweet_GRN_list) & set(ir94e_neurons)))}")

taste_types = ["Sweet", "Bitter", "Salty", "Water"]
orders_2 = [76, 64, 24, 56]
orders_3 = [1291, 1576, 870, 1133]

x = range(len(taste_types))
plt.bar(x, orders_2, width=0.4, label="2Ns")
plt.bar([i + 0.4 for i in x], orders_3, width=0.4, label="3Ns")
plt.xticks([i + 0.2 for i in x], taste_types)
plt.ylabel("Number of Neurons")
plt.title("2nd vs 3rd Order Neurons by Taste Modality")
plt.legend()
plt.show()

# bitter 395, lowsalt 221, sugar 514, water 323