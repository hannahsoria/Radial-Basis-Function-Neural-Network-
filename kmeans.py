'''kmeans.py
Performs K-Means clustering
Hannah Soria
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
import random
from matplotlib.colors import ListedColormap
from matplotlib import axes, cm
import sys
from matplotlib.animation import FuncAnimation



class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            self.num_samps, self.num_features = data.shape

        #variables for animation
        x_list = []
        self.x_list = x_list

        y_list = []
        self.y_list = y_list

        curr_index = 0
        self.curr_index = curr_index

        plot_data_x = []
        self.plot_data = plot_data_x

        plot_data_y = []
        self.plot_data = plot_data_y

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.num_features = data.shape[1]
        self.num_samps = data.shape[0]
        self.data = data

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''

        data_copy = np.copy(self.data)

        return data_copy

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        sum_square = np.sum(np.square(pt_1 - pt_2)) # finding sum of squares
        e_dist = np.sqrt(sum_square) # finding square root

        return e_dist

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        self.centroids = centroids #set centroids

        # math for euclidean distance
        temp = (np.square(pt - centroids)) 
        temp2 = (np.sum(temp, axis=1))
        e_dist = (np.sqrt(temp2))
        
        return e_dist

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        self.k = k
        self.num_features = self.data.shape[1]
        self.centroids = np.random.rand(k,self.num_features) # create random centroids

        return self.centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        self.initialize(k)
        curr_iter = 0 #current iteration
        while curr_iter < max_iter :
            curr_iter = curr_iter + 1 # increment the iterations
            self.update_labels(self.centroids) #update labels
            prev_label = self.centroids # keep track of the previous centroids
            new_cen, diff_cen = self.update_centroids(self.k, self.data_centroid_labels, prev_label) # update the centroids and track new centroids and the difference
            self.centroids = new_cen # update the centroids

            #mation for max difference
            abs_val = np.abs(diff_cen)
            max_diff = np.max(abs_val)
            if max_diff < tol:
                break

        self.compute_inertia()

        return self.inertia, curr_iter


    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        best_inertia = np.inf # infinity essentialy
        iters = [] # list for the inertias

        for j in range(n_iter): # for the number of times to run the k-means
            curr_inertia, i = self.cluster(k) # cluster
            iters.append(i) # add iteration to the list
            if curr_inertia < best_inertia: # if the inertia is the new lowest
                best_inertia = curr_inertia # set the inertia
                self.centroids = self.get_centroids() # set the centroids
                self.data_centroid_labels = self.get_data_centroid_labels() # sedt the labels
                self.inertia = best_inertia # set the inerita

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        if self.data_centroid_labels is None:
            self.data_centroid_labels = np.zeros(self.num_samps)
        for i in range(self.data.shape[0]):# loop through data
            vals = self.dist_pt_to_centroids(self.data[i,:], centroids) # keep distances
            min_val = np.argmin(vals) # find minimum of values
            self.data_centroid_labels[i] = min_val # set to the minimum
        self.data_centroid_labels = self.data_centroid_labels.astype(int)

        return self.data_centroid_labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        new_centroids = np.zeros(prev_centroids.shape) #prev_centroids.copy()

        for i in range(k): # for the umber of clusters
            values = self.data[data_centroid_labels == i, :] # if the label for the data equals
            if values.size == 0: # if there is no value
                new_centroid = self.data[np.random.randint(0, self.data.shape[0]), :] # create a random centroid
            else: 
                new_centroid = np.mean(values, axis=0) # the centroid is the mean
            new_centroids[i] = new_centroid # set the new centroids

        self.centroids = new_centroids # set the centroid
        centroid_diff = new_centroids - prev_centroids # find the differenece

        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        distance = 0
        # math to compute inertia
        for i in range(self.data.shape[0]):
            data = self.data[i] 
            centroid = self.dist_pt_to_centroids(data,self.centroids)
            min_cen = (np.min(centroid))
            min_cen = min_cen * min_cen
            distance += min_cen
            self.inertia = distance / self.data.shape[0]

        return self.inertia

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        for i in range(self.k):
            cluster = self.data[self.data_centroid_labels == i,:] # get the cluster
            colors = cm.get_cmap('Set1') # make cmap
            cmap1 = ListedColormap(colors(range(15))) # make cmap
            plt.scatter(cluster[:,0],cluster[:,1], cmap = cmap1) # plot the data
            plt.xlabel("X")
            plt.ylabel("Y")
            valk = str(self.k)
            val = "Kmeans for " + valk + " clusters"
            plt.title(val)

        for i in range(self.centroids.shape[0]): # plot the centroids
            plt.plot(self.centroids[i][0],self.centroids[i][1], '*',markersize=12,color = "black")
        plt.show()

    def elbow_plot(self, max_k): # n_iter is for the updated plot
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: number of iterations

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertia_list = []
        k_vals = [i for i in range(1,max_k + 1, 1)]
        for i in range(1, max_k + 1, 1):
            # for j in range(1, n_iter + 1, 1): #for the updated plot
            #     self.cluster_batch(i,j)#for the updated plot
            # inertia_list.append(self.inertia) # for updated plots
            inertia,_ = self.cluster(i) #for the original plot
            inertia_list.append(inertia) #for the original plot
        plt.figure(figsize=(16,8)) # figure
        plt.plot(k_vals,inertia_list, marker=".", markersize=20) # x = top PCs included y = proportion variance accounted for
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.title('The Elbow Method showing clusters vs. inertia')

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        for i in range(self.data.shape[0]):
            index = self.data_centroid_labels[i]
            self.data[i,:] = self.centroids[index.astype("int")] 

    # ANIMATION EXTENSION CODE 
    '''
    this is the code that I worked on to create an animated scatterplot. It was not entirely successful
    but this is the process: I attempted to do this through recursion. I had a main function that would 
    do find all of the values for x and put them into an array and the same for y. Then I created a new 
    list that would be for the coordinated to be plotted. The list first only contains the first corrdinates.
    Then I plotted those coordinates and call animation. This uses FuncAnimation which calls update which
    incremets the index of the lists. The idea behind this is that the list would then be incremented 
    through for every value and plotted with animation that way. This did not work because there are
    holes in the recursion logic and there are issues with collecting the correct data points for the 
    lists. I worked to debug these isses but was unsuccessful. I also looked towards other methods such
    as plt.pause() This would just print every point on a seperate plot. If I were to continue to work
    on animation I would figure out how to get the correct data then how to get the recursion to run.
    I don't fully understand how FuncAnimation works after reading the documentation but I would ask 
    for help or find more reading for it.
    '''
    def main_fun(self):
        # plot the centroids
        for i in range(self.centroids.shape[0]):
            plt.plot(self.centroids[i][0],self.centroids[i][1], '*',markersize=12,color = "black")
        # get the clusters
        for j in range(self.k):
            cluster = self.data[self.data_centroid_labels == j,:]
            # for every item in the cluster
            for c in range(cluster.shape[0]):
                #append the x value to the x list
                self.x_list.append(cluster[c,0])
                # apprend the y value to the y list
                self.y_list.append(cluster[c,1])
        fig = plt.figure()
        # x data to be plotted
        self.plot_data_x = [self.x_list[self.curr_index]]
        # y data to be plotted
        self.plot_data_y = [self.y_list[self.curr_index]]
        # plot the data
        plt.scatter(x=self.plot_data_x, y =self.plot_data_y)
        
        #animiation which calls updat
        animation = FuncAnimation(fig, self.update(), interval=2000, repeat = True)
        plt.show()
        animation

    # increment the index
    def update(self):
        self.curr_index += 1
