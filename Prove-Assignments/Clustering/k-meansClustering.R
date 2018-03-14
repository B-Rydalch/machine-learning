library(cluster)
library(datasets)
library(readr)
library(factoextra)

# 1. Load the Data
myData = state.x77
#View(myData)

######################################
# 2. Hierarchical Clustering 
#    and denogram
######################################

# first compute a distance matrix
distance = dist(as.matrix(myData))

# now perform the clustering
hc = hclust(distance)

# finally, plot the dendrogram
plot(hc)

######################################
# 3. Scaling data (normalizing)
#    plot and comparing differences
######################################
data_scaled = scale(myData)
colMeans(data_scaled)
distance = dist(as.matrix(data_scaled))
hc = hclust(distance)
plot(hc, xlab="States", ylab="Cluster")


######################################
# 4. Remove area column
######################################
df = subset(myData, select = -c(Area) )
distance = dist(as.matrix(df))
hc = hclust(distance)
plot(hc, xlab="States", ylab="Cluster")


######################################
# 5. cluster on the frost 
######################################
Frost <- myData[,7,drop=FALSE]
distance = dist(as.matrix(Frost))
hc = hclust(distance)
plot(hc, xlab="States", ylab="Cluster")

######################################
# k-means Clustering
# 1. use the normalized data
# 2. cluster into 3 clusters
######################################
k_clusters = kmeans(data_scaled, centers=3)
                    
# summary of the clusters
summary(k_clusters)

# Centers (mean values) of the clusters
k_clusters$centers

# Cluster assignments
k_clusters$cluster

######################################
# 3. loop for clustering sum of 
#    squares error for each k value. 
# 4. performing a elbow method
######################################
errorVector <- vector()
for (k in 1:25) {
  # calculate elbow diagram
  errorVector[k] <- kmeans(data, k)$tot.withinss
}

# Within-cluster sum of squares 
k_clusters$withinss

# tot.withinss is total sum of squares across clusters
k_clusters$tot.withinss

# plot the elbow diagram to see the amount of error
plot(errorVector, xlab="k", ylab="total within-cluster sum of squares error")

# view states in each cluster
summary(k_clusters)

######################################
# 6 Plotting a 2D visual representation of k-means clusters
######################################
clusplot(myData, k_clusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)


######################################
# 7 Analyze the centers of each of these clusters.
######################################
print(k_clusters$centers)

