import numpy
import math
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

class KmeansClustering(TimeSeriesKMeans):

    def __init__(self, num_clusters=5, random_state=23) -> None: 
        """This function initialises K-means clustering with DTW distance metric
        and DBA for computing centroids by setting number of clusters and random
        state provided.        

        :param num_clusters: Number of clusters for clustering, defaults to 5
        :type num_clusters: int, optional
        :param random_state: Random number generator used to initialize clustering, defaults to 23
        :type random_state: int, optional
        """

        super().__init__(num_clusters, metric="dtw",           
                          max_iter_barycenter=10,
                          random_state=random_state)               
        self.num_clusters = num_clusters                  

    def compute_clusters(self, data: numpy.ndarray) -> numpy.array:
        """This function fits DBA k-means to given data and
        predicts nearest cluster to which each time series in data belongs.

        :param data: Timeseries data of shape (n_ts_train, size, dimension).
        :type data: numpy.ndarray                  
        :return: Nearest cluster assigned for each time series in data. 
        """           
        print("DBA k-means")           

        self.data = data
        self.data_preds = self.fit_predict(self.data)        

    def visualize_clustering_results(self):
        """This function provides visualization of clustering performed on
        the given data and accepts predicted labels returned from function compute_clusters().

        :param y_preds: Predictions returned by compute_clusters() on data used for clustering.
        : type y_preds: numpy ndarray
        """
        self.sz = self.data.shape[1]

        plot_sz = math.ceil(self.num_clusters/2)

        for yi in range(self.num_clusters):
            plt.subplot(plot_sz, 2, 1 + yi)
            for xx in self.data[self.data_preds == yi]:
                plt.plot(xx.flatten(), "k-", alpha=.2)
            plt.plot(self.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, self.sz)
            if yi == 1:
                plt.title("DBA $k$-means")
  

    def get_cluster_statistics(self):
        mean = []
        cluster_std = []
        cluster_var = []

        for i in range(self.num_clusters):

            cluster = self.data[self.data_preds == i]
            cluster = numpy.array(cluster)
            
            mean.append(numpy.mean(cluster))

            cluster =  numpy.array(cluster)

            std = numpy.std(cluster, axis=1)
            var = numpy.var(cluster, axis=1)

            std = std.flatten()
            var = var.flatten()

            std_max = std.max()           
            var_max = var.max()

            cluster_std.append(std_max)
            cluster_var.append(var_max)                

        stats = pd.DataFrame({'mean': mean,
                                'standard deviation': cluster_std,
                                'variance': cluster_var

            })

        clusters = [i for i in range(self.num_clusters)]

        stats.insert(0, 'clusters', clusters)
        stats.set_index('clusters', inplace=True)

        return stats

    def get_cluster_centroid_statistics(self):

        mean = []
        std = []
        var = []


        for i in range(self.num_clusters):

            cluster = self.cluster_centers_[i]

            mean.append(numpy.mean(cluster))
            std.append(numpy.std(cluster))
            var.append(numpy.var(cluster, dtype=numpy.float64))            

        stats_centroid = pd.DataFrame({'centroid mean': mean,
                            'centroid standard deviation': std,
                            'centroid variance': var

        })

        clusters = [i for i in range(self.num_clusters)]

        stats_centroid.insert(0, 'clusters', clusters)
        stats_centroid.set_index('clusters', inplace=True)

        return stats_centroid

    def compute_mean_relative_change(self, testtimestamps, forecasts, res):
        mean = []
        std = []
        var = []
        cluster_assigned = []
        cluster_mean = []


        for i in testtimestamps:

            data = numpy.array(forecasts[i])

            mean.append(numpy.mean(data))
            std.append(numpy.std(data))
            var.append(numpy.var(data, dtype=numpy.float64))
            cluster_assigned.append(res[i])

            cluster_mean.append(numpy.mean(self.cluster_centers_[res[i]]))

        tests = pd.DataFrame({'test values': testtimestamps,
                            'mean': mean,
                            'standard deviation': std,
                            'variance': var,
                            'cluster_assigned': cluster_assigned,
                            'cluster_mean': cluster_mean

        })

        tests['mean diff'] = (tests['mean'] - tests['cluster_mean'])
        tests['mean relative change'] = tests['mean diff'] / abs(tests['cluster_mean'])

        self.tests_mrc = tests

        return self.tests_mrc

    def visualize_cluster_assignement_forecast(self, index, forecasts, res):

        forecast_cluster = res[index]
        forecast = forecasts[index]

        for xx in self.data[self.data_preds == forecast_cluster]:
            plt.plot(xx.flatten(), "k-", alpha=.2)
        plt.plot(self.cluster_centers_[forecast_cluster].ravel(), "r-")        
        plt.plot(forecast, "y-")
        plt.xlim(0, self.sz)

        return forecast_cluster, plt
   



