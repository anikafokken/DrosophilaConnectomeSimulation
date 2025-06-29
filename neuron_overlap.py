# Package imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# DataFrame variables
classifications = pd.read_csv("database\\classification_630.csv") # change this back sometime
connections = pd.read_csv("database\\connections_630.csv")
connections = connections[connections["syn_count"] >= 5]
labels = pd.read_csv("database\\labels.csv")

# From connectome paper data as of September 2024
sugar_GRNs = pd.read_csv("database\\sugar_GRNs.csv")
water_GRNs = pd.read_csv("database\\water_GRNs.csv")
bitter_GRNs = pd.read_csv("database\\bitter_GRNs.csv")
lowsalt_GRNs = pd.read_csv("database\\lowsalt_GRNs.csv")

sugar_2Ns = pd.read_csv("output_csvs\\sugar_2Ns.csv")['root_id']
water_2Ns = pd.read_csv("output_csvs\\water_2Ns.csv")['root_id']
bitter_2Ns = pd.read_csv("output_csvs\\bitter_2Ns.csv")['root_id']
lowsalt_2Ns = pd.read_csv("output_csvs\\lowsalt_2Ns.csv")['root_id']

tastes_2Ns = {
    "sugar": sugar_2Ns,
    "water": water_2Ns,
    "bitter": bitter_2Ns,
    "ir94e": lowsalt_2Ns
}

# pd.set_option("display.max_colwidth", None)

# Function definitions

# Finds total upstream neurons without considering order
def find_total_upstream_neurons(neuron_df, connections):
    # Define a 'visited' and 'frontier' list to keep track of upstream neurons that are already accounted for
    visited = set(neuron_df["post_root_id"])
    frontier = set(neuron_df["post_root_id"])
    while len(frontier) > 0:
        new_neurons = set()
        # Where the post root ID is the same as the post root ID from the input neuron list
        new_connections = connections[connections["post_root_id"].isin(frontier)]
        # Find the pre root IDs of the post root IDs (the roots upstream of each ID in the input list)
        upstream_neurons = set(new_connections["pre_root_id"])
        if new_neurons == 0:
            break  
        for n in upstream_neurons:
            if n not in visited:
                new_neurons.add(n) 
        # Add the new neurons of this loop into the visited list
        visited.update(new_neurons)
        frontier = new_neurons
    return list(visited)

# Finds total downstream neurons without considering order
def find_total_downstream_neurons(neuron_df, connections):
    # Define a 'visited' and 'frontier' list to keep track of downstream neurons that are already accounted for
    visited = set(neuron_df["pre_root_id"])
    frontier = set(neuron_df["pre_root_id"])
    while len(frontier) > 0:
        new_neurons = set()
        # Where the pre root ID is the same as the pre root ID from the input neuron list
        new_connections = connections[connections["pre_root_id"].isin(frontier)]
        # Find the post root IDs of the pre root IDs (the roots downstream of each ID in the input list)
        downstream_neurons = set(new_connections["post_root_id"])
        if new_neurons == 0:
            break  
        for n in downstream_neurons:
            if n not in visited:
                new_neurons.add(n)
        # Add the new neurons of this loop into the visited list
        visited.update(new_neurons)
        frontier = new_neurons
    return list(visited)

# Finds the total neurons, including upstream and downstream neurons but doesn't account for order (probably won't use)
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
    

# Merges a list of GRNs with a connections CSV to grab pre root IDs that are equivalent to the root IDs from the input list
def get_connections(neuron_list):
    return pd.merge(neuron_list, connections, left_on='root_id', right_on='pre_root_id', how='inner')

# Finds all neurons, including upstream and downstream neurons using the above functions, and finds all unique roots
def get_total_neurons(combined_neuron_list):
    upstream = find_total_upstream_neurons(combined_neuron_list, connections)
    downstream = find_total_downstream_neurons(combined_neuron_list, connections)
    combined_unique_neurons = list(set(upstream) | set(downstream))
    return combined_unique_neurons

# Finds shared upstream and downstream nodes of a given order between two lists of GRNs **Does consider order
def find_shared_nodes(neuron_list1, neuron_list2, merge1, merge2, order):
    # Merges with a connections CSV if needed (depending on merge boolean parameters)
    if merge1:
        combined_neuron1_list = get_connections(neuron_list1)
    else:
        combined_neuron1_list = neuron_list1
    if merge2:
        combined_neuron2_list = get_connections(neuron_list2)
    else:
        combined_neuron2_list = neuron_list2

    # Find n order neurons (based on argument, i.e. 2Ns or 3Ns) for each list of neurons
    n_order_neus1 = get_n_order_neurons(combined_neuron1_list, connections, order)
    n_order_neus2 = get_n_order_neurons(combined_neuron2_list, connections, order)
    
    # Finds shared neurons between the two lists of n order neurons
    shared_neurons = list(set(n_order_neus1) & set(n_order_neus2))
    return shared_neurons


# Finds all neurons of a given order (2 or 3)
def get_n_order_neurons(neu_list, connections, order, merge, taste):
    # Apply synapse threshold (5 for 2Ns, 10 for 3Ns)
    if order == 2:
        connections = connections[connections['syn_count'] >= 5]
    elif order == 3:
        connections = connections[connections['syn_count'] >= 10]

    # Merge GRN list with connections if needed (some input lists may be from connections CSVs, so they don't need to merge with the connections CSV)
    if merge:
        neu_list = get_connections(neu_list)

    # Define a 'visited' and 'frontier' list to keep track of upstream neurons that are already accounted for
    current_ids = set(neu_list['root_id'])
    visited = set(current_ids)

    for i in range(order-1):
        # Where the pre root ID is the same as the root ID from the input neuron list
        current_conns = connections[connections['pre_root_id'].isin(current_ids)]
        next_ids = set(current_conns['post_root_id'])

        # Optional: filter out GRNs or 2Ns of the same modality **only on last step and for finding 3Ns**
        if i == (order-2):
            # Get GRNs (gustatory + same taste type)
            grn_filter = (classifications["class"] == "gustatory") & (classifications["sub_class"] == taste)
            grn_ids = set(classifications[grn_filter]["root_id"])
            next_ids -= grn_ids

        # Remove already visited; set the current IDs to the ones discovered in this loop, so they can be explored for 3Ns if applicable
        next_ids -= visited
        visited.update(next_ids)
        current_ids = next_ids

    return current_ids

# Finds the total shared neurons between all four gustatory senses (sugar, water, bitter, salt) for a given order
def total_shared_order_neurons(sugar, water, bitter, salt, connections, order):
    # Build n_order neurons dictionary
    n_order_neurons = {
        "sweet": get_n_order_neurons(sugar, connections, order, True, "sugar"),
        "water": get_n_order_neurons(water, connections, order, True, "water"),
        "bitter": get_n_order_neurons(bitter, connections, order, True, "bitter"),
        "ir94e": get_n_order_neurons(salt, connections, order, True, "ir94e")
    }

    print(f"\n{order} Order Neuron Counts:")
    for taste, neurons in n_order_neurons.items(): # Print the get_n_order_neurons() results for each taste (each item in n_order_neurons dictionary)
        print(f"{taste}: {len(neurons)}")

    print(f"\nShared {order} Order Neurons:")
    for r in range(2, 5):  
        # Finds pairwise, 3-wise, 4-wise overlaps of each taste to exhaust all combinations between tastes
        for combo in combinations(n_order_neurons.keys(), r):
            # Finds shared neurons for each combo of tastes' neuron lists
            shared = set.intersection(*[n_order_neurons[taste] for taste in combo])
            print(f"{list(combo)} → {len(shared)} neurons shared")



# Finding total shared order neurons

total_shared_order_neurons(sugar_GRNs, water_GRNs, bitter_GRNs, lowsalt_GRNs, connections, 2)
total_shared_order_neurons(sugar_GRNs, water_GRNs, bitter_GRNs, lowsalt_GRNs, connections, 3)


# Finding total neurons of individual senses

# Gr5a (Sweet sensing)
sweet_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "sugar/water")]
find_print_total_neurons(sweet_GRN_list, "sweet", True)
# print(get_n_order_neurons(sweet_GRN_list, connections, 2, True))
print(f"2nd order neurons: {len(get_n_order_neurons(sugar_GRNs, connections, 2, True, "sugar"))}")
print(f"3rd order neurons: {len(get_n_order_neurons(sugar_GRNs, connections, 3, True, "sugar"))}")


bitter_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "bitter")]
find_print_total_neurons(bitter_GRNs, "bitter", True)
print(f"2nd order neurons: {len(get_n_order_neurons(bitter_GRNs, connections, 2, True, "bitter"))}")
print(f"3rd order neurons: {len(get_n_order_neurons(bitter_GRNs, connections, 3, True, "bitter"))}")


# Ir76b/Ir94e (Protein sensing/"umami")
salty_GRN_list = classifications[(classifications["class"] == "gustatory") & (classifications["flow"] != "intrinsic") & (classifications["sub_class"] == "low-salt")]
find_print_total_neurons(lowsalt_GRNs, "salty", True)
print(f"2nd order neurons: {len(get_n_order_neurons(lowsalt_GRNs, connections, 2, True, "ir94e"))}")
print(f"3rd order neurons: {len(get_n_order_neurons(lowsalt_GRNs, connections, 3, True, "ir94e"))}")


find_print_total_neurons(water_GRNs, "water", True)
print(f"2nd order neurons: {len(get_n_order_neurons(water_GRNs, connections, 2, True, "water"))}")
print(f"3rd order neurons: {len(get_n_order_neurons(water_GRNs, connections, 3, True, "water"))}")



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
# print(f"\n\nTotal shared nodes between Gr5a and Ir94e (including upstream/downstream nodes): {len(find_shared_nodes(sweet_GRN_list, ir94e_neurons, True, True))}")
# print(f"Shared neurons between Gr5a and Ir94e: {len(list(set(sweet_GRN_list) & set(ir94e_neurons)))}")


# Conclusion:

# bitter 395, lowsalt 221, sugar 514, water 323
# Ours: 412, 241, 550, 346 (average difference of +24)