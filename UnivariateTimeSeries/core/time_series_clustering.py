import numpy
import math
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

from core.ecdf_distance_measures import WassersteinDistance, CramerVonMisesDistance, KuiperDistance, AndersonDarlingDistance, KolmogorovSmirnovDistance, DTSDistance

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
                          max_iter_barycenter=50,
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

        plt.figure(figsize=(14,8))

        font = {'color':  'red',        
        'size': 10        
        }

        for yi in range(self.num_clusters):
            index = 1 + yi
            # plt.subplot(plot_sz, 3, index)
            if yi == 6:
                plt.subplot(plot_sz, 3, 8)
            else:                
                plt.subplot(4, 3, index)
            for xx in self.data[self.data_preds == yi]:
                plt.plot(xx.flatten(), "k-", alpha=.2)
            plt.plot(self.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, self.sz)            
            plt.text(0.50, 0.85,'Cluster %d' % (yi), fontdict=font, transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("DTW $k$-means clustering", fontsize=18, pad=16)
            if yi == 3:
                plt.ylabel(' Stock closing price', fontsize=14, family='monospace', labelpad=18, loc='bottom')
            #if yi == 4:
                #plt.xlabel('Time', fontsize=14, family='monospace', labelpad=18)                 
            
        
        plt.xlabel('Time', fontsize=14, family='monospace', labelpad=18)        
        # plt.savefig('clustering.jpg', dpi=500, bbox_inches='tight')
        # plt.savefig('clustering.eps', dpi=100, bbox_inches='tight')
        # plt.tight_layout()
        plt.show()


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

    def compute_mean_relative_change(self, timeindex, tests, res):
        mean = []
        std = []
        var = []
        cluster_assigned = []
        cluster_mean = []


        for i in timeindex:

            data = numpy.array(tests[i])

            mean.append(numpy.mean(data))
            std.append(numpy.std(data))
            var.append(numpy.var(data, dtype=numpy.float64))
            cluster_assigned.append(res[i])

            cluster_mean.append(numpy.mean(self.cluster_centers_[res[i]]))

        result = pd.DataFrame({'time index': timeindex,
                            'mean': mean,
                            'standard deviation': std,
                            'variance': var,
                            'cluster_assigned': cluster_assigned,
                            'cluster_mean': cluster_mean

        })

        result['mean diff'] = (result['mean'] - result['cluster_mean'])
        result['mean relative change'] = result['mean diff'] / abs(result['cluster_mean'])

        self.tests_mrc = result

        return self.tests_mrc

    def visualize_cluster_assignement_test(self, index, tests, res):

        test_cluster = res[index]
        test = tests[index]

        for xx in self.data[self.data_preds == test_cluster]:
            plt.plot(xx.flatten(), "k-", alpha=.2)
        plt.plot(self.cluster_centers_[test_cluster].ravel(), "r-")        
        plt.plot(test, "y-")
        plt.xlim(0, self.sz)

        return test_cluster, plt

    def compute_std_variance_relative_change(self, timeindex, tests, res):

        mean = []
        std_test = []
        var_test = []
        cluster_assigned = []
        cluster_mean = []        
        cluster_std = []
        cluster_var = []

        for i in timeindex:

            data = numpy.array(tests[i])

            mean.append(numpy.mean(data))
            std_test.append(numpy.std(data))
            var_test.append(numpy.var(data, dtype=numpy.float64))
            cluster_assigned.append(res[i])

            cluster_mean.append(numpy.mean(self.cluster_centers_[res[i]]))  
            cluster_centre = res[i]
            
            cluster = self.data[self.data_preds == cluster_centre]
            cluster =  numpy.array(cluster)

            std = numpy.std(cluster, axis=1)
            var = numpy.var(cluster, axis=1)

            std = std.flatten()
            var = var.flatten()

            std_max = std.max()
            var_max = var.max()

            cluster_std.append(std_max)
            cluster_var.append(var_max)

        tests = pd.DataFrame({'time index': timeindex,
                            'mean': mean,
                            'standard deviation': std_test,
                            'variance': var_test,
                            'cluster_assigned': cluster_assigned,
                            'cluster_mean': cluster_mean, 
                            'cluster_std': cluster_std,
                            'cluster_var': cluster_var

        })

        tests['max std diff'] = (tests['standard deviation'] - tests['cluster_std'])
        tests['std reltive change'] = tests['max std diff'] / abs(tests['cluster_std'])

        tests['max var diff'] = (tests['variance'] - tests['cluster_var'])
        tests['var reltive change'] = tests['max var diff'] / abs(tests['cluster_var'])

        return tests

    def _get_statistical_dist_measures(self, X1, X2):   

        CVM_distance = CramerVonMisesDistance().compute_distance(X1, X2)
        Anderson_Darling_distance = AndersonDarlingDistance().compute_distance(X1, X2)
        Kolmogorov_Smirnov_distance = KolmogorovSmirnovDistance().compute_distance(X1, X2)
        Kuiper_distance = KuiperDistance().compute_distance(X1, X2)
        Wasserstein_distance = WassersteinDistance().compute_distance(X1, X2)
        DTS_distance = DTSDistance().compute_distance(X1, X2)   
        
        return {'Anderson_Darling_dist': Anderson_Darling_distance,
                'CVM_dist': CVM_distance,
                'DTS_dist':DTS_distance,
                'Kolmogorov_Smirnov_dist':Kolmogorov_Smirnov_distance,
                'Kuiper_dist': Kuiper_distance,
                'Wasserstein_dist': Wasserstein_distance}

    def ecdf_between_cluster_and_data(self, t, scaler, tests, res, clusters_wd_dist):        
        mean = []        
        cluster_assigned = []    
        distances_all = []
        wd_dist = []

        for i in t:

            data = numpy.array(tests[i])

            mean.append(numpy.mean(data))    
            cluster_assigned.append(res[i])
            wd_dist.append(clusters_wd_dist[res[i]])

            preds_inv = scaler.inverse_transform(data)
            cluster_inv = scaler.inverse_transform(self.cluster_centers_[res[i]])
            
            distances = self._get_statistical_dist_measures(cluster_inv.flatten(), preds_inv.flatten())

            distances_all.append(distances)

        tests = pd.DataFrame({'Test point': t,
                            'Test point mean': mean,                   
                            'Assigned cluster': cluster_assigned,
                            'WD origin': wd_dist                   

        })
        distances_df = pd.DataFrame(distances_all)

        result = pd.concat([tests, distances_df], axis=1)

        return result    

    def compute_wasserstein_distance(self, X1, X2):        
        return WassersteinDistance().compute_distance(X1, X2)




