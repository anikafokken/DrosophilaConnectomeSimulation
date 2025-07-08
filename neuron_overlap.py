# Package imports
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from venn import venn

# DataFrame variables
classifications = pd.read_csv("database\\classification.csv") # change this back sometime
connections = pd.read_csv("database\\connections_630.csv") # version 630
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

sugar_3Ns = pd.read_csv("output_csvs\\sugar_3Ns.csv")['root_id']
water_3Ns = pd.read_csv("output_csvs\\water_3Ns.csv")['root_id']
bitter_3Ns = pd.read_csv("output_csvs\\bitter_3Ns.csv")['root_id']
lowsalt_3Ns = pd.read_csv("output_csvs\\lowsalt_3Ns.csv")['root_id']

tastes_2Ns = {
    "sugar": sugar_2Ns,
    "water": water_2Ns,
    "bitter": bitter_2Ns,
    "ir94e": lowsalt_2Ns
}

# pd.set_option("display.max_colwidth", None)

# Function definitions

# Finds total upstream neurons without considering order
def find_total_upstream_neurons(combined_neuron_df, connections):
    # Define a 'visited' and 'frontier' list to keep track of upstream neurons that are already accounted for
    visited = set(combined_neuron_df["post_root_id"])
    frontier = set(combined_neuron_df["post_root_id"])
    while len(frontier) > 0:
        new_neurons = set()
        # Where the post root ID is the same as the post root ID from the input neuron list
        new_connections = connections[connections["post_root_id"].isin(frontier)]
        # Find the pre root IDs of the post root IDs (the roots upstream of each ID in the input list)
        upstream_neurons = set(new_connections["pre_root_id"])
        new_neurons = upstream_neurons - visited
        if len(new_neurons) == 0:
            break  
        # Add the new neurons of this loop into the visited list
        visited.update(new_neurons)
        frontier = new_neurons
    return list(visited)

# Finds total downstream neurons without considering order
def find_total_downstream_neurons(combined_neuron_df, connections):
    # Define a 'visited' and 'frontier' list to keep track of downstream neurons that are already accounted for
    visited = set(combined_neuron_df["pre_root_id"])
    frontier = set(combined_neuron_df["pre_root_id"])
    while len(frontier) > 0:
        new_neurons = set()
        # Where the pre root ID is the same as the pre root ID from the input neuron list
        new_connections = connections[connections["pre_root_id"].isin(frontier)]
        # Find the post root IDs of the pre root IDs (the roots downstream of each ID in the input list)
        downstream_neurons = set(new_connections["post_root_id"])
        new_neurons = downstream_neurons - visited
        if len(new_neurons) == 0:
            break
        # Add the new neurons of this loop into the visited list
        visited.update(new_neurons)
        frontier = new_neurons
    return list(visited)   

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
        neu_list = get_connections(neu_list)

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
                    (classifications["class"] == "gustatory") |
                    (classifications["root_id"].isin(twoNs_dfs[taste])) # TODO: is this the problem? they filter by no GRNS of any modality, and no 2Ns of the same modality; issue is likely ir94e
                )
                grn_ids = set(classifications[grn_filter]["root_id"])
                next_ids -= grn_ids

        # Remove already visited; set the current IDs to the ones discovered in this loop, so they can be explored for 3Ns if applicable
        next_ids -= visited
        visited.update(next_ids)
        current_ids = next_ids
    
    return current_ids


def get_df_ids_length(df):
    return len(df["root_id"])

def total_shared_order_neurons(sugar, water, bitter, salt, connections, order):
    second_order = {
        "sugar": get_n_order_neurons(sugar, connections, order, True, "sugar"),
        "water": get_n_order_neurons(water, connections, order, True, "water"),
        "bitter": get_n_order_neurons(bitter, connections, order, True, "bitter"),
        "ir94e": get_n_order_neurons(salt, connections, order, True, "ir94e")
    }

    for r in range(2, 5):  # combinations of size 2 to 4
        for combo in combinations(second_order.keys(), r):
            # print(set(second_order[combo[0]]) & set(second_order[combo[1]]))
            shared_neurons = set.intersection(*[second_order[name] for name in combo])
            print(f"Total shared neurons in {list(combo)}: {len(shared_neurons)}")

def get_by_nt(neu_df, nt):
    return neu_df[neu_df['nt_type'] == nt]

def find_nt_percentages(neu_df):
    GLUT = len(get_by_nt(neu_df, "GLUT")['root_id'])
    GABA = len(get_by_nt(neu_df, "GABA")['root_id'])
    ACH = len(get_by_nt(neu_df, "ACH")['root_id'])
    print(f"GLUT: {round(GLUT/len(neu_df['root_id'])*100, 2)}%\nGABA: {round(GABA/len(neu_df['root_id'])*100, 2)}%\nACH: {round(ACH/len(neu_df['root_id'])*100, 2)}%")

def composite_report(neu_df1, neu_df2, neu_df3, neu_df4, merge=True, filtering_on=True):
    dfs = {
        "sugar": neu_df1,
        "bitter": neu_df2,
        "ir94e": neu_df3,
        "water": neu_df4
    }
    for taste in dfs:

        if merge:
            combined_neuron_list = get_connections(dfs[taste])
        else:
            combined_neuron_list = dfs[taste]

        neurons_count = len(combined_neuron_list["pre_root_id"].unique())
        upstream = find_total_upstream_neurons(combined_neuron_list, connections)
        downstream = find_total_downstream_neurons(combined_neuron_list, connections)
        combined_unique_neurons = get_total_neurons(combined_neuron_list)
        
        print(f"\n\nNumber of {taste} neurons = {neurons_count}")
        print("Upstream neurons: ", len(upstream) - neurons_count)
        print("Downstream neurons: ", len(downstream) - neurons_count)
        print(f"Total {taste}-associated neurons: {len(combined_unique_neurons)}")
        print(f"2nd order neurons: {len(get_n_order_neurons(dfs[taste], connections, 2, True, taste, filtering_on))}")
        print(f"3rd order neurons: {len(get_n_order_neurons(dfs[taste], connections, 3, True, taste, filtering_on))}")
        find_nt_percentages(get_connections(dfs[taste]))

    total_shared_order_neurons(neu_df1, neu_df2, neu_df3, neu_df4, connections, 2)
    total_shared_order_neurons(neu_df1, neu_df2, neu_df3, neu_df4, connections, 3)


# Error checking
def find_discrepancies(my_neu_list, threeNs_list, classifications):
    sugar_3Ns = pd.DataFrame(threeNs_list, columns=['root_id'])
    paper_classifications = pd.merge(sugar_3Ns, classifications, left_on='root_id', right_on='root_id', how='inner')
    print(f"Paper classifications: {paper_classifications}")
    my_sugar_3Ns = pd.DataFrame(list(get_n_order_neurons(sugar_GRNs, connections, 3, True, "sugar")), columns=['root_id'])
    print("My classifications: ", pd.merge(my_sugar_3Ns, classifications, on='root_id'))
    my_sugar_3Ns = my_neu_list
    my_sugar_3Ns = set(my_sugar_3Ns['root_id'])
    sugar_3Ns = set(sugar_3Ns['root_id'])
    not_shared = pd.DataFrame(sugar_3Ns ^ my_sugar_3Ns, columns=['root_id'])
    not_shared_classifications = pd.merge(not_shared, classifications, left_on='root_id', right_on='root_id', how='inner')
    # print(sugar_3Ns ^ my_sugar_3Ns)
    print(not_shared_classifications)



composite_report(sugar_GRNs, water_GRNs, bitter_GRNs, lowsalt_GRNs, True, filtering_on=True)


# Conclusion (3Ns)

# Theirs: bitter 395, ir94e 221, sugar 514, water 323
# Ours without filtering: 412, 241, 550, 346 (average difference of +24)

# Ours with filtering: 343, 194, 461, 297