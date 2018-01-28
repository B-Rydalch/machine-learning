import numpy as np

class kNeighborsClassifier():
    def __init__(self, k = 3):
        self.k = k

    def fit(self, data, targets):
        return KNeighborsModel(self.k,data,targets)

class KNeighborsModel(object):
    def __init__(self,k,data, targets):
        self.k = k
        self.data = data
        self.targets = targets

    def predict(self, data_test):
        return self.knn(self.k, self.data, self.targets, data_test)

    def knn(self, k, data, targets, data_test):
        nInputs = np.shape(data_test)[0]
        closest = np.zeros(nInputs) 

        for n in range(nInputs):
            
            #compute distances
            distance = np.sum((data - data_test[n,:])**2, axis = 1)

            #idendtify the nearest neighbours 
            indices = np.argsort(distance, axis = 0)

            classes = np.unique(targets[indices[:k]])

         #   print"classes:", classes
            

            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(k):
                    counts[targets[indices[i]]] += 1
                closest[n] = np.max(counts)

            return closest

