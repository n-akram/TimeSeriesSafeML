import numpy
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

class KmeansClustering:

    def __init__(self, num_clusters=5, random_state=23) -> None: 
        """This function initialises K-means clustering with DTW distance metric
        and DBA for computing centroids by setting number of clusters and random
        state provided.        

        :param num_clusters: Number of clusters for clustering, defaults to 5
        :type num_clusters: int, optional
        :param random_state: Random number generator used to initialize clustering, defaults to 23
        :type random_state: int, optional
        """               
        self.numClusters = num_clusters
        self.dba_km = TimeSeriesKMeans(n_clusters=self.numClusters,                          
                          metric="dtw",                          
                          max_iter_barycenter=10,
                          random_state=random_state)          

    def compute_clusters(self, data: numpy.ndarray) -> numpy.array:
        """This function fits DBA k-means to given data and
        predicts nearest cluster to which each time series in data belongs.

        :param data: Timeseries data of shape (n_ts_train, size, dimension).
        :type data: numpy.ndarray                  
        :return: Nearest cluster assigned for each time series in data. 
        """           
        print("DBA k-means")           

        self.data = data
        self.data_preds = self.dba_km.fit_predict(self.data)

        return self.data_preds

    def visualize_clustering_results(self, y_preds):
        """This function provides visualization of clustering performed on
        the given data and accepts predicted labels returned from function compute_clusters().

        :param y_preds: Predictions returned by compute_clusters() on data used for clustering.
        : type y_preds: numpy ndarray
        """
        sz = self.data.shape[1]

        for yi in range(self.numClusters):
            plt.subplot(4, 2, 1 + yi)
            for xx in self.data[y_preds == yi]:
                plt.plot(xx.flatten(), "k-", alpha=.2)
            plt.plot(self.dba_km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            if yi == 1:
                plt.title("DBA $k$-means")        