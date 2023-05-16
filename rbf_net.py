'''rbf_net.py
Radial Basis Function Neural Network
HANNAH SORIA
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import kmeans


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

        self.k = num_hidden_units

        self.num_classes = num_classes

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        distance = np.zeros(centroids.shape[0]) #empty array to hold the distances

        for i in range(centroids.shape[0]): #for all the data in centroids
            values = data[cluster_assignments == i,:] # all the values that are of the same cluster assginment
            dist = [] # list for the distances 
            for j in values: # for every data point in the cluster
                d = kmeans_obj.dist_pt_to_pt(j, centroids[i]) # the distance form data to centroid
                dist.append(d) # append to the list
            distance[i] = np.mean(dist) # find the mean and put it in the array 

        return distance

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        kmean_obj = kmeans.KMeans(data)
        kmean_obj.cluster_batch(k = self.k, n_iter = 10, verbose = False)
        self.prototypes = kmean_obj.get_centroids()
        cluster_assigns = kmean_obj.get_data_centroid_labels()
        sigs = self.avg_cluster_dist(data, self.prototypes, cluster_assigns, kmean_obj)
        self.sigmas = sigs


    def linear_regression(self, A, y):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''
        ones = np.ones((len(A),1)) # col of ones
        Ahat = np.hstack([A, ones]) # make Ahat
        slope_coeffs = np.linalg.lstsq(Ahat, y, rcond=None)[0]

        return slope_coeffs

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        k_mean_obj = kmeans.KMeans(data)
        activation = np.zeros((data.shape[0], self.k))

        for i in range(data.shape[0]):
            dists = k_mean_obj.dist_pt_to_centroids(data[i,:],self.get_prototypes()) #distance between data sample and the prototype of the hidden unit
            dists = dists ** 2 #square the distance
            denom = 2 * self.sigmas ** 2 + 0.000001 #sigma squared then times 2 + epsilon
            num = dists/denom #divide the two
            activation[i,:] = np.exp(-num)#all negative and exponential function

        return activation
    
    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        bias = np.ones((hidden_acts.shape[0],1)) # create a column of ones
        hidden = np.hstack((hidden_acts, bias)) # stack the ones for the bias
        output = hidden@self.wts # multiply the hidden activation by the weights

        return output

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        self.initialize(data) #initializing the network
        activations = self.hidden_act(data) # getting the activation for the data
        class_identifier = np.zeros((y.shape[0], self.num_classes)) # if index at y is of the class currently view = 1 otherwise 0
        weights = np.zeros((self.k+1, self.num_classes)) #for the weights including the bias 

        for i in range(self.num_classes): #for all of the output classes
            for j in range(0, y.shape[0]): #for each class
                if y[j] == i: # if the class is the same
                    class_identifier[j][i] = 1 # set to 1
                else:
                    class_identifier[j][i] = 0 # set to 0

        for i in range(self.num_classes): # for all the output classes
            weight = self.linear_regression(activations, class_identifier[:,i]) # do linear regression for each class int he output
            weights[:,i] = weight #set the weights

        self.wts = weights # update weights


    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        y_pred = np.zeros(data.shape[0]) # create empty array for predictions
        activations = self.hidden_act(data) # get the activations
        outputs = self.output_act(activations) # get the outputs for the activations

        for i in range(outputs.shape[0]): # for all of the outputs
            max_val = np.argmax(outputs[i]) # retrieve the max value for the row
            y_pred[i] = max_val # set the prediction tot he max value

        return y_pred

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        accuracy = np.sum(y == y_pred) / y.shape[0]

        return accuracy

    # EXTENSIONS
    
    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        num_classes = len(np.unique(y)) # total number of classes
        con_matrix = np.zeros((num_classes,num_classes)) # empty confusion matrix

        for actual, predicted in zip(y, y_pred): # iterate through y and y pred getting the actual and predicted at the same time
            con_matrix[actual][predicted] += 1 
        
        return con_matrix
    
