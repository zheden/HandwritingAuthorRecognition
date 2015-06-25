import random
import numpy as np
import matplotlib.pyplot as plt
import operator

# Euclidean Distance between two vectors
def euclideanDist(x,y):
	sumSq=0.0
	#add up the squared differences
	for i in range(len(x)):
		sumSq+=(x[i]-y[i])**2
	#take the square root of the result
	return (sumSq**0.5)

def calculateCentroid(i_centroidPixel, i_WRITERS_COUNT, i_descriptors):
    X = []
    Y = []
    for i in range(i_WRITERS_COUNT):
        sumX = 0
        sumY = 0
        for j in range(len(i_descriptors[i])):
            sumX += i_descriptors[i][j][0]
            sumY += i_descriptors[i][j][1]
            X.append(i_descriptors[i][j][0])
            Y.append(i_descriptors[i][j][1])
        i_centroidPixel[i][0] = sumX/len(i_descriptors[i])
        i_centroidPixel[i][1] = sumY/len(i_descriptors[i])
    #plt.plot(X, Y, 'ro')
    #plt.show()
    return i_centroidPixel

def getMajority(i_neighbors):
	classVotes = {}
	for x in range(len(i_neighbors)):
		response = i_neighbors[x][0]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getNeighbors(i_testPoint, i_trainDescriptors, i_k):
    distances = []
    for x in range(len(i_trainDescriptors)):
        #print i_trainDescriptors[x]
        for y in range(len(i_trainDescriptors[x])):
            dist = euclideanDist(i_testPoint, i_trainDescriptors[x][y])
            distances.append((x+1, i_trainDescriptors[x][y], dist))
    #print distances
    distances.sort(key=operator.itemgetter(2))
    #print distances
    neighbors = []
    for x in range(i_k):
        neighbors.append(distances[x])
    return neighbors

def main():
    ########### create sample input ###############
    DESCRIPTORS_COUNT = 5
    WRITERS_COUNT = 4
    descriptors = {} # replace with original descriptors
    descriptors = {k: [] for k in range(WRITERS_COUNT)}
    for i in range(WRITERS_COUNT):
        for j in range(DESCRIPTORS_COUNT):
            descriptors[i].append([random.uniform(0.5+1*i, 1.0+1*i), random.uniform(0.5+1*i, 1.0+1*i)])
    print descriptors
    ##################################################

    # calculate centroid of each cluster of descriptors
    centroidPixel = {}
    centroidPixel = {k: [0,0] for k in range(WRITERS_COUNT)}
    centroidPixel = calculateCentroid(centroidPixel, WRITERS_COUNT, descriptors)

    # new point to be categorized in a cluster
    testPoint = [3.5,4.3]
    euclideanDistance = 10000000
    clusterNumber = 0
    for i in range(len(centroidPixel)):
        dist = euclideanDist(testPoint, centroidPixel[i]) # calculate Euclidean Distance of testPoint to centroids of every cluster
        #print dist
        if euclideanDistance > dist:
            euclideanDistance = dist
            clusterNumber = i+1
    #print euclideanDistance
    print "According to centroid method, test point belongs to user : " + str(clusterNumber)

    k = 3
    neighbors = getNeighbors(testPoint, descriptors, k) # Get k nearest neighbors
    response = getMajority(neighbors) # Get id of neighbor who won majority votes
    print "According to centroid method, test point belongs to user : " + str(response)

if __name__ == "__main__":
    main()