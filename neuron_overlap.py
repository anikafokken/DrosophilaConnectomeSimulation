# Package imports
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from venn import venn

global connections
# DataFrame variables
classifications = pd.read_csv("database\\classification_630.csv") # change this back sometime
conn = pd.read_csv("database\\connections_630.csv") # version 630
# connections = connections[connections["syn_count"] >= 5]
labels = pd.read_csv("database\\labels.csv")

# From connectome paper data as of September 2024
sugar_GRNs = pd.read_csv("database\\sugar_GRNs.csv")
water_GRNs = pd.read_csv("database\\water_GRNs.csv")
bitter_GRNs = pd.read_csv("database\\bitter_GRNs.csv")
ir94e_GRNs = pd.read_csv("database\\ir94e_GRNs.csv")
lowsalt_GRNs = pd.read_csv("database\\lowsalt_GRNs.csv")
sour_GRNs = pd.read_csv("database\\sour_GRNs.csv")


sugar_2Ns = pd.read_csv("output_csvs\\sugar_2Ns.csv")['root_id']
water_2Ns = pd.read_csv("output_csvs\\water_2Ns.csv")['root_id']
bitter_2Ns = pd.read_csv("output_csvs\\bitter_2Ns.csv")['root_id']
lowsalt_2Ns = pd.read_csv("output_csvs\\lowsalt_2Ns.csv")['root_id']
ir94e_2Ns = pd.read_csv("output_csvs\\ir94e_2Ns.csv")['root_id']



sugar_3Ns = pd.read_csv("output_csvs\\sugar_3Ns.csv")['root_id']
water_3Ns = pd.read_csv("output_csvs\\water_3Ns.csv")['root_id']
bitter_3Ns = pd.read_csv("output_csvs\\bitter_3Ns.csv")['root_id']
lowsalt_3Ns = pd.read_csv("output_csvs\\lowsalt_3Ns.csv")['root_id']

tastes_2Ns = {
    "sugar": sugar_2Ns,
    "water": water_2Ns,
    "bitter": bitter_2Ns,
    "ir94e": ir94e_2Ns
}

all_GRNs = pd.concat([sugar_GRNs, water_GRNs,bitter_GRNs,lowsalt_GRNs], axis=0)
plt.rcParams["font.family"] = "Times New Roman"

def second_order(taste_GRNs, taste):
        
        # Get connectivity from GRNs
        connectivity = pd.merge(taste_GRNs['root_id'], conn[['pre_root_id','post_root_id','neuropil','syn_count','nt_type']], left_on = 'root_id', right_on = 'pre_root_id', how = 'inner').query("syn_count >= 5")
        connectivity = connectivity.drop(columns = 'root_id')
        
        # Aggregate this connectivity to get 2Ns
        second_orders = connectivity.groupby("post_root_id").agg({'pre_root_id' : 'nunique', 'syn_count' : 'sum'}).reset_index()
        second_orders.columns = ['root_id','upstream_' + taste + '_GRNs', taste + '_syn_count']
        second_orders['const'] = 1 # this will be needed for the OLS regression later
        
        return connectivity, second_orders

sugar_second_order_connectivity, sugar_2Ns = second_order(sugar_GRNs, 'sugar')
print(sugar_second_order_connectivity)
water_second_order_connectivity, water_2Ns = second_order(water_GRNs, 'water')
bitter_second_order_connectivity, bitter_2Ns = second_order(bitter_GRNs, 'bitter')
lowsalt_second_order_connectivity, lowsalt_2Ns = second_order(lowsalt_GRNs, 'lowsalt')

def third_order(taste_second_order_connectivity, taste, taste_second_orders):
            connectivity = pd.merge(taste_second_order_connectivity.query("syn_count >= 10")['post_root_id'], conn[['pre_root_id','post_root_id','neuropil','syn_count','nt_type']],
                                    left_on = 'post_root_id', right_on = 'pre_root_id', how = 'inner').query("syn_count >= 10")
            print('Min_c:', min(connectivity['syn_count']))
            print(len(taste_second_order_connectivity))
            print(len(sugar_2Ns))
            connectivity = connectivity.drop(columns = 'post_root_id_x')
            connectivity.rename(columns={'post_root_id_y': 'post_root_id'}, inplace=True)
            GRNs = pd.concat([sugar_GRNs['root_id'],  water_GRNs['root_id'], bitter_GRNs['root_id'],lowsalt_GRNs['root_id']])
            print('Connectivity', len(connectivity))
            
            connectivity = connectivity[~connectivity['pre_root_id'].isin(GRNs)] # drop entries where GRN is presynaptic
            print(len(connectivity))
            connectivity = connectivity[~connectivity['post_root_id'].isin(GRNs)] # drop if GRN is postsynaptic
            print(len(connectivity))
            connectivity = connectivity[~connectivity['post_root_id'].isin(taste_second_orders['root_id'])] # drop if 2N of same modality is postsynaptic
            print('With filters:', len(connectivity))
            connectivity = connectivity.drop_duplicates()
            print('Dropped duplicates', len(connectivity))
            third_orders = connectivity.groupby("post_root_id").agg({'pre_root_id' : 'nunique', 'syn_count' : 'sum'}).reset_index()
            print('third orders: ', len(third_orders))
            print('Min_d:', min(connectivity['syn_count']))

            third_orders.columns = ['root_id','upstream_' + taste + '_2Ns', taste + '_syn_count']
            third_orders['const'] = 1  # this will be used for OLS regression
            return connectivity, third_orders

sugar_third_order_connectivity, sugar_3Ns = third_order(sugar_second_order_connectivity, 'sugar', sugar_2Ns)
water_third_order_connectivity, water_3Ns = third_order(water_second_order_connectivity, 'water', water_2Ns)
bitter_third_order_connectivity, bitter_3Ns = third_order(bitter_second_order_connectivity, 'bitter', bitter_2Ns)
lowsalt_third_order_connectivity, lowsalt_3Ns = third_order(lowsalt_second_order_connectivity, 'lowsalt', lowsalt_2Ns)

print(np.unique(sugar_3Ns.root_id.values).shape)
print(np.unique(water_3Ns.root_id.values).shape)
print(np.unique(bitter_3Ns.root_id.values).shape)
print(np.unique(lowsalt_3Ns.root_id.values).shape)


sugar_list = {'{}'.format(value) for value in sugar_2Ns['root_id'].unique()}
sugar_dict = {'Sugar 2Ns': sugar_list}

water_list = {'{}'.format(value) for value in water_2Ns['root_id'].unique()}
water_dict = {'Water 2Ns': water_list}

bitter_list = {'{}'.format(value) for value in bitter_2Ns['root_id'].unique()}
bitter_dict = {'Bitter 2Ns': bitter_list}

lowsalt_list = {'{}'.format(value) for value in lowsalt_2Ns['root_id'].unique()}
lowsalt_dict = {'IR94e 2Ns': lowsalt_list}

colors = ('#cf4848','orange','#3489eb','purple')
crossover = {**sugar_dict, **water_dict, **bitter_dict, **lowsalt_dict}
venn(crossover, cmap = ListedColormap(colors), figsize = (8,8), fontsize = 14)



sugar_list = {'{}'.format(value) for value in sugar_3Ns['root_id'].unique()}
sugar_dict = {'Sugar 3Ns': sugar_list}

water_list = {'{}'.format(value) for value in water_3Ns['root_id'].unique()}
water_dict = {'Water 3Ns': water_list}


bitter_list = {'{}'.format(value) for value in bitter_3Ns['root_id'].unique()}
bitter_dict = {'Bitter 3Ns': bitter_list}

lowsalt_list = {'{}'.format(value) for value in lowsalt_3Ns['root_id'].unique()}
lowsalt_dict = {'IR94e 3Ns': lowsalt_list}

colors = ('#cf4848','orange','#3489eb','purple')
crossover = {**sugar_dict, **water_dict, **bitter_dict, **lowsalt_dict}
venn(crossover, cmap = ListedColormap(colors), figsize = (8,8), fontsize = 14)

# pd.set_option("display.max_colwidth", None)

# Function definitions

# Finds total upstream neurons without considering order
def find_total_upstream_neurons(combined_neuron_df, connections):
    # Define a 'visited' and 'frontier' list to keep track of upstream neurons that are already accounted for
    visited = set(combined_neuron_df["post_root_id"])
    frontier = set(combined_neuron_df["post_root_id"])
    times = 0
    while len(frontier) > 0:
        times += 1
        new_neurons = set()
        # Where the post root ID is the same as the post root ID from the input neuron list
        new_connections = connections[connections["post_root_id"].isin(frontier)]
        # Find the pre root IDs of the post root IDs (the roots upstream of each ID in the input list)
        upstream_neurons = set(new_connections["pre_root_id"])
        new_neurons = upstream_neurons - visited
        if len(new_neurons) == 0:
            break  
        # Add the new neurons of this loop into the visited list
        frontier = new_neurons
        print("\n", times, len(visited), len(frontier))
        visited.update(new_neurons)
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
        filtered_connections = new_connections[new_connections['syn_count'] > 5]
        print(new_connections['neuropil'].unique())
        # Find the post root IDs of the pre root IDs (the roots downstream of each ID in the input list)
        downstream_neurons = set(filtered_connections["post_root_id"])
        new_neurons = downstream_neurons - visited
        if len(new_neurons) == 0:
            break
        # Add the new neurons of this loop into the visited list
        visited.update(new_neurons)
        frontier = new_neurons
    return list(visited)   

# Merges a list of GRNs with a connections CSV to grab pre root IDs that are equivalent to the root IDs from the input list
def get_connections(neuron_list):
    return pd.merge(neuron_list, conn, left_on='root_id', right_on='pre_root_id', how='inner')

# Finds all neurons, including upstream and downstream neurons using the above functions, and finds all unique roots
def get_total_neurons(combined_neuron_list):
    upstream = find_total_upstream_neurons(combined_neuron_list, conn)
    downstream = find_total_downstream_neurons(combined_neuron_list, conn)
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
    n_order_neus1 = get_n_order_neurons(combined_neuron1_list, conn, order, True, conn)
    n_order_neus2 = get_n_order_neurons(combined_neuron2_list, conn, order, True, conn)
    
    # Finds shared neurons between the two lists of n order neurons
    shared_neurons = list(set(n_order_neus1) & set(n_order_neus2))
    return shared_neurons


# Finds all neurons of a given order (2 or 3)
def get_n_order_neurons(neu_list, order, taste, filtering_on, connections):
    # print(connections)
    
    taste_3Ns = []
    
    sub_class_terms = {
                "sugar": "sugar/water",
                "bitter": "bitter",
                "ir94e": "low-salt",
                "water": "sugar/water"
            }
    twoNs_dfs = {
                "sugar": sugar_2Ns,
                "bitter": bitter_2Ns,
                "ir94e": ir94e_2Ns,
                "water": water_2Ns
                # "sour": sour_2Ns
            }
    
    # Apply synapse threshold (5 for 2Ns, 10 for 3Ns)
    if order == 2:
        connections = connections[connections['syn_count'] >= 5]
    elif order == 3:
        connections = connections[connections['syn_count'] >= 10]
        print("Min", min(connections['syn_count']))
        print(connections)

    # Merge GRN list with connections if needed (some input lists may be from connections CSVs, so they don't need to merge with the connections CSV)
    
    
    def real_third_order(order, neu_list, taste, connections, i):
        if order == 3 and taste != "sour":
            GRNs = pd.concat([sugar_GRNs['root_id'],  water_GRNs['root_id'], bitter_GRNs['root_id'],lowsalt_GRNs['root_id']])

            if taste == 'ir94e': taste = 'lowsalt'
            taste_second_order_connectivity, taste_2Ns = second_order(neu_list, taste)
            print(taste_second_order_connectivity)
            connectivity = pd.merge(taste_second_order_connectivity.query('syn_count >= 10')['post_root_id'], connections, left_on='post_root_id', right_on='pre_root_id', how='inner').query('syn_count >= 10')
            connectivity = connectivity.drop(columns='post_root_id_x')
            connectivity.rename(columns={'post_root_id_y': 'post_root_id'}, inplace=True)
            print(len(taste_2Ns))
            print(connectivity)
            print(len(connectivity))
            connectivity = connectivity[~connectivity['pre_root_id'].isin(GRNs)] # drop entries where GRN is presynaptic
            print(len(connectivity))
            connectivity = connectivity[~connectivity['post_root_id'].isin(GRNs)] # drop if GRN is postsynaptic
            print(len(connectivity))
            connectivity = connectivity[~connectivity['post_root_id'].isin(taste_2Ns['root_id'])] # drop if 2N of same modality is postsynaptic
            print(len(connectivity))
            connectivity = connectivity.drop_duplicates()
            print(len(connectivity))
            third_orders = connectivity.groupby("post_root_id").agg({'pre_root_id' : 'nunique', 'syn_count' : 'sum'}).reset_index()
            
            print(taste, connectivity, len(connectivity), len(third_orders))
            print(f"returning {i}")
            taste_3Ns.append(third_orders['post_root_id'])
            return third_orders['post_root_id']
    
    raw_neu_list = neu_list
    neu_list = get_connections(neu_list)

    # Define a 'visited' and 'frontier' list to keep track of upstream neurons that are already accounted for
    current_ids = set(neu_list['root_id'])
    visited = set(current_ids)

    print(f"Shared in order {order}", current_ids & set(twoNs_dfs['sugar']['root_id']))

    for i in range(order-1):
        print("i:", i)
        if i == 1:
            thirds = real_third_order(order, raw_neu_list, taste, connections, i)
            if thirds is not None:
                print(f"returning {i}", thirds)
                print("all thirds:", taste_3Ns)
                return thirds
    #     # Where the pre root ID is the same as the root ID from the input neuron list
        if order == 2:
            current_conns = connections[connections['pre_root_id'].isin(current_ids)]
            next_ids = set(current_conns['post_root_id'])

            if filtering_on:
                if order == 2:
                    # print(order) # TODO: filter out all GRNs
                    print("Next:", len(next_ids))
                    
                    grn_ids = set(classifications[classifications['root_id'].isin(next_ids) & classifications['root_id'].isin(all_GRNs)]['root_id'])
                    print("Filtered:", len(grn_ids))
                    next_ids -= grn_ids
                if order == 3:
                    
                    # Get GRNs (gustatory + same taste type)
                    if taste != "sour":
                        grn_filter = (
                            (classifications["class"] == "gustatory") |
                            (classifications["root_id"].isin(twoNs_dfs[taste])) # TODO: is this the problem? they filter by no GRNS of any modality, and no 2Ns of the same modality; issue is likely ir94e
                        )
                        
                        # print(taste, grn_filter)
                        grn_ids = set(classifications[grn_filter]["root_id"])
                        print("Filtered:", len(next_ids - grn_ids))
                        next_ids -= grn_ids

            # Remove already visited; set the current IDs to the ones discovered in this loop, so they can be explored for 3Ns if applicable
            if order == 2: next_ids -= visited
            print("Visited: ", len(visited))
            for neuron in next_ids:
                if neuron in all_GRNs:
                    print("in all GRNs")
            visited.update(next_ids)
            current_ids = next_ids
            print(len(next_ids))
        
    return current_ids


sugar_2Ns = pd.DataFrame(get_n_order_neurons(sugar_GRNs, 2, "sugar", True, conn), columns=['root_id'])
water_2Ns = pd.DataFrame(get_n_order_neurons(water_GRNs, 2, "water", True, conn), columns=['root_id'])
bitter_2Ns = pd.DataFrame(get_n_order_neurons(water_GRNs, 2, "bitter", True, conn), columns=['root_id'])
ir94e_2Ns = pd.DataFrame(get_n_order_neurons(ir94e_GRNs, 2, "ir94e", True, conn), columns=['root_id'])
sour_2Ns = pd.DataFrame(get_n_order_neurons(sour_GRNs, 2, "sour", True, conn), columns=['root_id'])



def get_df_ids_length(df):
    return len(df["root_id"])

def total_shared_order_neurons(sugar, water, bitter, ir94e, conn, order):

    sugar_downstream = get_n_order_neurons(sugar, 3, "sugar", True, conn)
    water_downstream = get_n_order_neurons(water, 3, "water", True, conn)
    bitter_downstream = get_n_order_neurons(bitter, 3, "bitter", True, conn)
    ir94e_downstream = get_n_order_neurons(ir94e, 3, "ir94e", True, conn)


    print("*****************************************************")
    print(len(sugar_downstream))
    print(len(water_downstream))
    print(len(bitter_downstream))
    print(len(ir94e_downstream))
    sugar_downstream = pd.DataFrame(list(get_n_order_neurons(sugar, order, "sugar", True, conn)), columns=['root_id'])
    water_downstream = pd.DataFrame(list(get_n_order_neurons(water, order, "water", True, conn)), columns=['root_id'])
    bitter_downstream = pd.DataFrame(list(get_n_order_neurons(bitter, order, "bitter", True, conn)), columns=['root_id'])
    ir94e_downstream = pd.DataFrame(list(get_n_order_neurons(ir94e, order, "ir94e", True, conn)), columns=['root_id'])
    # sour_2Ns = pd.DataFrame(get_n_order_neurons(sour_GRNs, order, "sour", True, conn), columns=['root_id'])
   
    print(len(sugar_downstream))
    print(len(water_downstream))
    print(len(bitter_downstream))
    print(len(ir94e_downstream))

    print(sugar_downstream)
    print(water_downstream)
    print(bitter_downstream)
    print(ir94e_downstream)

    sugar_list = {'{}'.format(value) for value in sugar_downstream['root_id'].unique()}
    sugar_dict = {f'Sugar {order}Ns': sugar_list}

    water_list = {'{}'.format(value) for value in water_downstream['root_id'].unique()}
    water_dict = {f'Water {order}Ns': water_list}

    bitter_list = {'{}'.format(value) for value in bitter_downstream['root_id'].unique()}
    bitter_dict = {f'Bitter {order}Ns': bitter_list}

    lowsalt_list = {'{}'.format(value) for value in ir94e_downstream['root_id'].unique()}
    lowsalt_dict = {f'IR94e {order}Ns': lowsalt_list}

    crossover = {**sugar_dict, **water_dict, **bitter_dict, **lowsalt_dict}
    print(crossover)
    plt.rcParams["font.family"] = "Times New Roman"
    venn(crossover)
    
    sets2 = {
        "Sugar": set(sugar_downstream),
        "Water": set(water_downstream),
        "Bitter": set(bitter_downstream),
        "Ir94e": set(ir94e_downstream)
    }

    sets3 = {
        "Sugar": set(sugar_3Ns),
        "Water": set(water_3Ns),
        "Bitter": set(bitter_3Ns),
        "Ir94e": set(lowsalt_3Ns)
    }

    second_order_calculated = {
        "sugar": get_n_order_neurons(sugar, 2, "sugar", True, conn),
        "water": get_n_order_neurons(water, 2, "water", True, conn),
        "bitter": get_n_order_neurons(bitter, 2, "bitter", True, conn),
        "ir94e": get_n_order_neurons(ir94e, 2, "ir94e", True, conn)
    }

    third_order_calculated = {
        "sugar": get_n_order_neurons(sugar, 3, "sugar", True, conn),
        "water": get_n_order_neurons(water, 3, "water", False, conn),
        "bitter": get_n_order_neurons(bitter, 3, "bitter", True, conn),
        "ir94e": get_n_order_neurons(ir94e, 3, "ir94e", True, conn)
    }

    # only_in_theirs = set(bitter_3Ns) - set(third_order_calculated["bitter"])
    # only_in_ours = set(third_order_calculated["bitter"]) - set(bitter_3Ns)
    # print(f"Only in theirs:", only_in_theirs)
    # print("Only in ours:", only_in_ours)


    # plotted_shared_neurons = {}
    # for r in range(2, 5):  # combinations of size 2 to 5
    #     for combo in combinations(next_order.keys(), r):
    #         # print(set(second_order[combo[0]]) & set(second_order[combo[1]]))
    #         shared_neurons = set.intersection(*[next_order[name] for name in combo])
    #         print(f"Total shared neurons in {list(combo)}: {len(shared_neurons)}")
    
    # if order == 2:
    #     venn(sets2)
    #     plt.title("Shared 2Ns")
    # elif order == 3:
    #     venn(third_order_calculated)
    #     plt.title("Shared 3Ns")

def get_by_nt(neu_df, nt):
    return neu_df[neu_df['nt_type'] == nt]

def get_nt_percent(neu_df, nt_name):
    nt = len(get_by_nt(neu_df, nt_name)['root_id'])
    nt_percentage = round(nt/len(neu_df['root_id'])*100, 2)
    return nt_percentage

def get_nt_percent_change(source_df, target_df, nt_name, i):
    if i == 1:
        source_nt_amount = 0

    else:
        source_nt_amount = len(get_by_nt(source_df, nt_name)['root_id'])
        target_nt_amount = len(get_by_nt(target_df, nt_name)['root_id'])
        percent_change = round(100*((target_nt_amount-source_nt_amount)/source_nt_amount), 2)

    return percent_change

def get_nt_amount(neu_df, nt_name):
    nt_amount = len(get_by_nt(neu_df, nt_name)['root_id'])
    return nt_amount

def find_nt_percentages(neu_df):
    GLUT = len(get_by_nt(neu_df, "GLUT")['root_id'])
    GABA = len(get_by_nt(neu_df, "GABA")['root_id'])
    ACH = len(get_by_nt(neu_df, "ACH")['root_id'])
    OCT = len(get_by_nt(neu_df, "OCT")['root_id'])
    SER = len(get_by_nt(neu_df, "SER")['root_id'])
    DA = len(get_by_nt(neu_df, "DA")['root_id'])
    nt_names = ["GLUT", "GABA", "ACH", "OCT", "SER", "DA"]
    for nt_i in range(len([GLUT, GABA, ACH, OCT, SER, DA])):
        nt_name = nt_names[nt_i]
        nt = [GLUT, GABA, ACH, OCT, SER, DA][nt_i]
        nt_percentage = get_nt_percent(neu_df, nt_name)
        print(f"{nt_name}: {nt_percentage}%")

def analyze_location(neu_df):
    downstream_neurons = conn[conn['post_root_id'].isin(find_total_downstream_neurons(neu_df, conn))]
    location_dict = {}
    for location in downstream_neurons['neuropil']:
        location_dict[location] = location_dict.get(location, 0) + 1
    plt.figure()
    plt.pie(location_dict.values(), explode=0, labels=location_dict.keys(), colors=None)
    return location_dict

# def GRN_heatmap(neu_dfs):
#     df = [sugar_GRNs, water_GRNs, bitter_GRNs, lowsalt_GRNs]
#     taste_connectivity = []
#     array = []
#     for i in range(len())

def composite_report(taste_dict, filtering_on=True):


    # bitter_4Ns = pd.Series(list(get_n_order_neurons(sugar_3Ns, 2, "sugar", True, conn)), name='root_id')
    # bitter_5Ns = pd.Series(list(get_n_order_neurons(sugar_4Ns, 2, "sugar", True, conn)), name='root_id')
    dfs = {
        "sugar": sugar_GRNs,
        "bitter": bitter_GRNs,
        "ir94e": ir94e_GRNs,
        "water": water_GRNs,
        "sour": sour_GRNs
    }

    taste_3Ns = []
    
    print(taste_dict['sugar'][0])

    nt_df = {}
    # for taste in dfs_special:
    #     nt_df[taste] = {}
    #     for nt in ['GABA', 'GLUT', 'ACH', 'SER', 'DA', 'OCT']:
    #         for layer_df in taste:
    #             nt_df[taste][nt] = 
    print(nt_df)
        
    
    colors = {
        "sugar": "red",
        "bitter": "green",
        "ir94e": "purple",
        "water": "blue",
        "sour": "orange"
    }
    
    for taste in taste_dict:
        combined_neuron_list = get_connections(dfs[taste])

        neurons_count = len(combined_neuron_list["pre_root_id"].unique())
        upstream = find_total_upstream_neurons(combined_neuron_list, conn)
        downstream = find_total_downstream_neurons(combined_neuron_list, conn)
        combined_unique_neurons = get_total_neurons(combined_neuron_list)
        
        print(f"\n\nNumber of {taste} neurons = {neurons_count}")
        print("Upstream neurons: ", len(upstream) - neurons_count)
        print("Downstream neurons: ", len(downstream) - neurons_count)
        print(f"Total {taste}-associated neurons: {len(combined_unique_neurons)}")
        print(f"2nd order neurons: {len(get_n_order_neurons(dfs[taste], 2, taste, filtering_on, conn))}")
        print(f"3rd order neurons: {len(get_n_order_neurons(dfs[taste], 3, taste, filtering_on, conn))}")

    plt.figure()
    for taste in taste_dict:
        x_points = list(range(len(taste_dict[taste])))
        y_points = [get_nt_percent(get_connections(taste_dict[taste][i]), "DA") for i in x_points]
        plt.plot(x_points, y_points, label=taste.capitalize(), color=colors[taste])
        
    plt.xlabel("Neuron Index")
    plt.ylabel("DA Percent")
    plt.title("DA Percent Across Neurons")
    plt.legend()

    plt.figure()
    for taste in taste_dict:
        x_points = list(range(len(taste_dict[taste])))
        y_points = [get_nt_percent(get_connections(taste_dict[taste][i]), "OCT") for i in x_points]
        plt.plot(x_points, y_points, label=taste.capitalize(), color=colors[taste])
        
    plt.xlabel("Neuron Index")
    plt.ylabel("OCT Percent")
    plt.title("OCT Percent Across Neurons")
    plt.legend()

    plt.figure()
    for taste in taste_dict:
        x_points = list(range(len(taste_dict[taste])))
        y_points = [get_nt_percent(get_connections(taste_dict[taste][i]), "SER") for i in x_points]
        plt.plot(x_points, y_points, label=taste.capitalize(), color=colors[taste])
        
    plt.xlabel("Neuron Index")
    plt.ylabel("SER Percent")
    plt.title("SER Percent Across Neurons")
    plt.legend()

    plt.figure()
    for taste in taste_dict:
        x_points = list(range(len(taste_dict[taste])))
        y_points = [get_nt_percent(get_connections(taste_dict[taste][i]), "GABA") for i in x_points]
        plt.plot(x_points, y_points, label=taste.capitalize(), color=colors[taste])
        
    plt.xlabel("Neuron Index")
    plt.ylabel("GABA Percent")
    plt.title("GABA Percent Across Neurons")
    plt.legend()




    # Amount-based
    plt.figure()
    for taste in taste_dict:
        x_points = list(range(len(taste_dict[taste])))
        y_points = [get_nt_amount(get_connections(taste_dict[taste][i]), "DA") for i in x_points]
        plt.plot(x_points, y_points, label=taste.capitalize(), color=colors[taste])
        
    # TODO: change these labels later to be more professional and accurate
    plt.xlabel("Neuron Index")
    plt.ylabel("DA Amount")
    plt.title("DA Amount Across Neurons")
    plt.legend()

    plt.figure()
    for taste in taste_dict:
        x_points = list(range(len(taste_dict[taste])))
        y_points = [get_nt_amount(get_connections(taste_dict[taste][i]), "OCT") for i in x_points]
        plt.plot(x_points, y_points, label=taste.capitalize(), color=colors[taste])
        
    plt.xlabel("Neuron Index")
    plt.ylabel("OCT Amount")
    plt.title("OCT Amount Across Neurons")
    plt.legend()

    plt.figure()
    for taste in taste_dict:
        x_points = list(range(len(taste_dict[taste])))
        y_points = [get_nt_amount(get_connections(taste_dict[taste][i]), "SER") for i in x_points]
        plt.plot(x_points, y_points, label=taste.capitalize(), color=colors[taste])
        
    plt.xlabel("Neuron Index")
    plt.ylabel("SER Amount")
    plt.title("SER Amount Across Neurons")
    plt.legend()

    plt.figure()
    for taste in taste_dict:
        x_points = list(range(len(taste_dict[taste])))
        y_points = [get_nt_amount(get_connections(taste_dict[taste][i]), "GABA") for i in x_points]
        plt.plot(x_points, y_points, label=taste.capitalize(), color=colors[taste])
        
    plt.xlabel("Neuron Index")
    plt.ylabel("GABA Amount")
    plt.title("GABA Amount Across Neurons")
    plt.legend()

    total_shared_order_neurons(sugar_GRNs, water_GRNs, bitter_GRNs, ir94e_GRNs, conn, 2)
    total_shared_order_neurons(sugar_GRNs, water_GRNs, bitter_GRNs, ir94e_GRNs, conn, 3)


# Error checking
def find_discrepancies(my_neu_list, threeNs_list, classifications):
    sugar_3Ns = pd.DataFrame(threeNs_list, columns=['root_id'])
    paper_classifications = pd.merge(sugar_3Ns, classifications, left_on='root_id', right_on='root_id', how='inner')
    print(f"Paper classifications: {paper_classifications}")
    my_sugar_3Ns = pd.DataFrame(list(get_n_order_neurons(sugar_GRNs, 3, "sugar", True, conn)), columns=['root_id'])
    print("My classifications: ", pd.merge(my_sugar_3Ns, classifications, on='root_id'))
    my_sugar_3Ns = my_neu_list
    my_sugar_3Ns = set(my_sugar_3Ns['root_id'])
    sugar_3Ns = set(sugar_3Ns['root_id'])
    not_shared = pd.DataFrame(sugar_3Ns ^ my_sugar_3Ns, columns=['root_id'])
    not_shared_classifications = pd.merge(not_shared, classifications, left_on='root_id', right_on='root_id', how='inner')
    # print(sugar_3Ns ^ my_sugar_3Ns)
    print(not_shared_classifications)

def analyze_downstream_neuropil(neu_list, taste):
    downstream_learning_rids = []
    for l in range(5):
        next_order_ids = get_n_order_neurons(neu_list, 2, taste, True, conn)
        learning_neurons = conn[conn['neuropil'] == 'MB']
        matched = learning_neurons[learning_neurons['post_root_id'].isin(next_order_ids)]
        print(matched)
        downstream_learning_rids.append(matched)
    print(f"Learning IDs: {downstream_learning_rids}")

# sour_2Ns = pd.Series(get_n_order_neurons(sour_GRNs, 2, "sour", True, conn), name='root_id')['root_id']

# Using the paper's code: 
sugar_third_order_connectivity, sugar_3Ns = third_order(sugar_second_order_connectivity, 'sugar', sugar_2Ns)
print("UNION:", len(set(sugar_3Ns['root_id']) & set(get_n_order_neurons(sugar_GRNs, 3, 'sugar', True, conn))))

recursive_3Ns = set(get_n_order_neurons(sugar_GRNs, 3, 'sugar', True, conn))
merged_3Ns = set(sugar_3Ns["root_id"])
shared = recursive_3Ns & merged_3Ns
only_recursive = recursive_3Ns - merged_3Ns
only_merged = merged_3Ns - recursive_3Ns

print(conn[conn['post_root_id'].isin(only_recursive)])
print(conn[conn['post_root_id'].isin(only_merged)])

print(f"Shared: {len(shared)}")
print(f"Only in recursive method: {len(only_recursive)}")
print(f"Only in merged method: {len(only_merged)}")

print(len(get_n_order_neurons(sugar_GRNs, 2, 'sugar', False, conn)))

print(sugar_2Ns)



sugar_4Ns = pd.Series(list(get_n_order_neurons(sugar_3Ns, 2, "sugar", True, conn)), name='root_id')
sugar_5Ns = pd.Series(list(get_n_order_neurons(sugar_4Ns, 2, "sugar", True, conn)), name='root_id')
sugar_6Ns = pd.Series(list(get_n_order_neurons(sugar_5Ns, 2, "sugar", True, conn)), name='root_id')

bitter_4Ns = pd.Series(list(get_n_order_neurons(bitter_3Ns, 2, "bitter", True, conn)), name='root_id')
bitter_5Ns = pd.Series(list(get_n_order_neurons(bitter_4Ns, 2, "bitter", True, conn)), name='root_id')
bitter_6Ns = pd.Series(list(get_n_order_neurons(bitter_5Ns, 2, "bitter", True, conn)), name='root_id')

ir94e_4Ns = pd.Series(list(get_n_order_neurons(lowsalt_3Ns, 2, "ir94e", True, conn)), name='root_id')
ir94e_5Ns = pd.Series(list(get_n_order_neurons(ir94e_4Ns, 2, "ir94e", True, conn)), name='root_id')
ir94e_6Ns = pd.Series(list(get_n_order_neurons(ir94e_5Ns, 2, "ir94e", True, conn)), name='root_id')

water_4Ns = pd.Series(list(get_n_order_neurons(water_3Ns, 2, "water", True, conn)), name='root_id')
water_5Ns = pd.Series(list(get_n_order_neurons(water_4Ns, 2, "water", True, conn)), name='root_id')
water_6Ns = pd.Series(list(get_n_order_neurons(water_5Ns, 2, "water", True, conn)), name='root_id')

print(type(sour_GRNs))
sour_2Ns = pd.Series(list(get_n_order_neurons(sour_GRNs, 2, "sour", False, conn)), name='root_id')
print(type(sour_2Ns))
sour_3Ns = pd.Series(list(get_n_order_neurons(sour_2Ns, 2, "sour", False, conn)), name='root_id')
sour_4Ns = pd.Series(list(get_n_order_neurons(sour_3Ns, 2, "sour", False, conn)), name='root_id')
sour_5Ns = pd.Series(list(get_n_order_neurons(sour_4Ns, 2, "sour", False, conn)), name='root_id')
sour_6Ns = pd.Series(list(get_n_order_neurons(sour_5Ns, 2, "sour", False, conn)), name='root_id')

taste_neuron_lists = {
    "sugar": [sugar_GRNs, sugar_2Ns, sugar_3Ns, sugar_4Ns, sugar_5Ns, sugar_6Ns],
    "bitter": [bitter_GRNs, bitter_2Ns, bitter_3Ns, bitter_4Ns, bitter_5Ns, bitter_6Ns],
    "ir94e": [ir94e_GRNs, ir94e_2Ns, lowsalt_3Ns, ir94e_4Ns, ir94e_5Ns, ir94e_6Ns],
    "water": [water_GRNs, water_2Ns, water_3Ns, water_4Ns, water_5Ns, water_6Ns],
    "sour": [sour_GRNs, sour_2Ns, sour_3Ns, sour_4Ns, sour_5Ns, sour_6Ns]
}



# print(connections['nt_type'].unique())
composite_report(taste_neuron_lists, filtering_on=True)
analyze_downstream_neuropil(sugar_GRNs, "sugar")
# analyze_location(sugar_GRNs)




plt.show()
# Conclusion (3Ns)

# Theirs: bitter 395, ir94e 221, sugar 514, water 323
# Ours without filtering: 412, 241, 550, 346 (average difference of +24)

# Ours with filtering: 343, 194, 461, 297