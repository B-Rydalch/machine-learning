import math
import numpy as np

# from group work
# calculates entropy and information gain
def calculate_entropy(classes, target):
    the_set = np.unique(classes)
    no_bin = 0.0
    yes_bin = 0.0
    total_entropy = 0.0

    for x in the_set:
        for y in range(len(classes)):
            if the_set[x] == classes[y]:
                if target[y] == 0:
                    no_bin += 1
                else:
                    yes_bin += 1

        total = no_bin + yes_bin
        no_bin_entropy = entropy(no_bin/total)
        yes_bin_entropy = entropy(yes_bin/total)
        single_entropy = no_bin_entropy + yes_bin_entropy
        weighted_entropy = single_entropy * (total / len(classes))
        total_entropy += weighted_entropy
        no_bin = yes_bin = 0.0

    return total_entropy
