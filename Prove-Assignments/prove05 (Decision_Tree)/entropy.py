import numpy as np
from node import Node
import operator

def entropy(p):
    if p!=0:
        return -p * np.log2(p)
    else:
        return 0

def most_common(train_data):
    count = {}
    unique, counts = np.unique(train_data, return_counts=True)
    count = dict(zip(unique, counts))
    most_common =max(count.iteritems(), key=operator.itemgetter(1))[0]
    
    return most_common

def print_tree(node):
    
    print ("node.name, ")
    if not node.isLeaf():
        for (key, value) in node.children.iteritems():
            print (key, "{",
            print_tree(value),
            print "}")

def build_tree(train_data, attributes, removed = []):
    # start by making an empty node
    current_node = Node()
    
    # remove any used attributes
    remaining = set(attributes) - set(removed)

    # test for same targets
    remaining_targets = train_data.iloc[:,-1].unique()

    # check to see if there are more rows
    if len(train_data) == 0:
        current_node.name = "Error no data"
        return current_node

    if len(remaining_targets) == 1:
        current_node.name = remaining_targets[0]
        return current_node
    
    #  when there are no more options
    elif len(remaining) == 0:
       
        #count the number of each class and return the most common
        leaf_value = Node(most_common(train_data.iloc[:,-1]))
        current_node.appendChild(leaf_value, leaf_value)
       
        return current_node
   
    else:
       
        # calculate the best value
        entropies = {}
        for attribute in remaining:
            entropies[attribute] = calculate_entropy(train_data, attribute)

        # find the lowest entropy weight in the data file
        best_value = min(entropies, key=entropies.get)
        current_node.name = best_value

        # get all possible values of root
        poss_values = train_data[best_value].unique()

        # build the tree
        child_nodes = {}
        for possible_value in poss_values:
            
             data_subset = train_data[train_data[best_value] == possible_value]
            data_subset.reset_index(inplace=True, drop=True)
            
            # remove this attribute
            removed.append(best_value)
            node = build_tree(data_subset, attributes, removed)
            
            # print "POS: ", possible_value
            child_nodes[possible_value] = node
            current_node.children = child_nodes
            removed = []

    return current_node

def calculate_entropy(train_data, attribute): 
    
    the_set = train_data[attribute].unique()
    no_bin = 0.0
    yes_bin = 0.0
    total_entropy = 0.0
    size = len(train_data)
    
    for attr in the_set:
        for x in range(0, size):
            if attr == train_data[attribute][x]:
                if train_data.iloc[:,-1][x] == 0:
                    no_bin += 1
                else:
                    yes_bin += 1

        total = no_bin + yes_bin
        no_bin_entropy = entropy(no_bin/total)
        yes_bin_entropy = entropy(yes_bin/total)
        single_entropy = no_bin_entropy + yes_bin_entropy
        weighted_entropy = single_entropy * (total / size)
    
        #print weighted_entropy
        total_entropy += weighted_entropy
        no_bin = yes_bin = 0.0
    
    return total_entropy