import math
import numpy as np




# From text book
def calc_entropy(p):
    if p!=0:
        return -p *np.log2(p)
    else:
        return 0


# From text book
def calc_info_gain(data,classes,feature):
    gain = 0
    nData = len(data)

    #List the values that feature can take
    values = []
    for datapoint in data:
        if datapoint[feature] not in values:
            values.append(datapoint[feature])

    featureCounts = np.zeros(len(values))
    entropy = np.zeros(len(values))
    valueIndex = 0

    #find where those values appear in data[feature] and the corresponding classes
    for value in values:
        dataIndex = 0
        newClasses = []

        for datapoint in data:
            if datapoint[feature]==value:
                featureCounts[valueIndex]+=1
                newClasses.append(classes[dataIndex])
            dataIndex += 1

        # Get the values in newClasses
        classValues = []
        for aclass in newClasses:
            if classValues.count(aclass)==0:
                classValues.append(aclass)
                classCounts = np.zeros(len(classValues))
                classIndex = 0
                for classValues in classValues:
                    for aclass in newClasses:
                        if aclass == classValue:
                            classCounts[classIndex]+=1
                    classInex += 1

                for classIndex in range(len(classValues))
                classIndex = 0
                for classValue in classValues:
                    for aclass in newClasses:
                        if aclass == classValue:
                            classCounts[classIndex]+=1
                    classIndex += 1

                for classIndex in range(len(classValues)):
                    entropy[valueIndex] += calc_entropy(float(classCounts[classIndex])/sum(classCounts))
            gain += float(featureCounts[valueIndex])/nData * entropy[valueIndex]
            valueIndex += 1
        return gain


# from group work
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
