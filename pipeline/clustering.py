import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import time


class ClusterAnalysis:
    """
    A class to perform feature selection using cluster analysis based on correlation coefficients.

    Attributes:
    -----------
    correlation_threshold : float
        The threshold for the correlation coefficient to form clusters.
    clusters : dict
        A dictionary to store the clusters and their selected features.
    selected_features : dict
        A dictionary to store the selected features for each cluster.

    Methods:
    --------
    _compute_correlation_matrix(df):
        Computes the correlation matrix for the given DataFrame.
    _hierarchical_clustering(corr_matrix):
        Performs hierarchical clustering on the given correlation matrix.
    _select_feature(X_train, X_test, X_valid, y_train, y_test, y_valid, features, method):
        Selects a feature from a cluster based on the specified method.
    _calculate_gini(X, y, feature):
        Calculates the Gini coefficient for a given feature.
    _select_by_gini(X, y, features):
        Selects a feature with the highest Gini coefficient.
    _select_by_gini_difference(X_train, X_test, y_train, y_test, features):
        Selects a feature with the smallest difference in Gini coefficients between train and test sets.
    _select_center_feature(features):
        Selects the feature closest to the center of the cluster.
    fit_transform(df, input_cols, selection_type):
        Fits the model to the data and transforms the input DataFrame.
    get_clusters():
        Returns the clusters formed after fitting the model.
    plot_dendrogram():
        Plots a dendrogram of the features and highlights the selected features.
    """

    def __init__(self, correlation_threshold):
        
        """
        Initializes the ClusterAnalysis class with the specified correlation threshold.

        Parameters:
        -----------
        correlation_threshold : float
            The threshold for the correlation coefficient to form clusters.
        """

        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.clusters = None

    def _compute_correlation_matrix(self, df):
        
        """
        Computes the correlation matrix for the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing the features.

        Returns:
        --------
        pd.DataFrame
            The correlation matrix.
        """

        return df.corr()

    def _hierarchical_clustering(self, corr_matrix):
        """
        Performs hierarchical clustering on the given correlation matrix.

        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            The correlation matrix.

        Returns:
        --------
        np.ndarray
            Cluster labels for each feature.
        """

        dist_matrix = 1 - np.abs(corr_matrix)
        
        # Ensure the distance matrix does not contain NaN or infinite values
        if np.any(np.isnan(dist_matrix)) or np.any(np.isinf(dist_matrix)):
            raise ValueError("The distance matrix contains NaN or infinite values.")
        
        condensed_dist_matrix = squareform(dist_matrix, checks=False)
        self.linkage_matrix = linkage(condensed_dist_matrix, method='average')  # ward can be used
        cluster_labels = fcluster(self.linkage_matrix, self.correlation_threshold, criterion='distance', depth=2)
        return cluster_labels

    def _select_feature(self, X_train, X_test, X_valid, y_train, y_test, y_valid, features, method):
        """
        Selects a feature from a cluster based on the specified method.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training data features.
        X_test : pd.DataFrame
            Test data features.
        X_valid : pd.DataFrame
            Validation data features.
        y_train : pd.Series
            Training data target.
        y_test : pd.Series
            Test data target.
        y_valid : pd.Series
            Validation data target.
        features : list
            List of features in the cluster.
        method : str
            Method to use for feature selection ('max_train', 'max_test', 'max_valid', 
                                                 'closest_train_test', 'center_cluster').

        Returns:
        --------
        str
            Selected feature from the cluster.
        """

        if method == 'max_train':
            return self._select_by_gini(X_train, y_train, features)
        elif method == 'max_test':
            return self._select_by_gini(X_test, y_test, features)
        elif method == 'max_valid':
            return self._select_by_gini(X_valid, y_valid, features)
        elif method == 'closest_train_test':
            return self._select_by_gini_difference(X_train, X_test, y_train, y_test, features)
        elif method == 'center_cluster':
            return self._select_center_feature(features)

    def _calculate_gini(self, X, y, feature):
        """
        Calculates the Gini coefficient for a given feature.

        Parameters:
        -----------
        X : pd.DataFrame
            Data features.
        y : pd.Series
            Data target.
        feature : str
            The feature for which to calculate the Gini coefficient.

        Returns:
        --------
        float
            Gini coefficient for the feature.
        """

        clf = LogisticRegression(random_state=42)
        clf.fit(X[[feature]].fillna(0), y)
        y_pred_prob = clf.predict_proba(X[[feature]].fillna(0))[:, 1]
        gini = 2 * roc_auc_score(y, y_pred_prob) - 1
        return gini

    def _select_by_gini(self, X, y, features):
        """
        Selects a feature with the highest Gini coefficient.

        Parameters:
        -----------
        X : pd.DataFrame
            Data features.
        y : pd.Series
            Data target.
        features : list
            List of features in the cluster.

        Returns:
        --------
        str
            Feature with the highest Gini coefficient.
        """

        gini_scores = {}
        for feature in features:
            gini_scores[feature] = self._calculate_gini(X, y, feature)
        return max(gini_scores, key=gini_scores.get)

    def _select_by_gini_difference(self, X_train, X_test, y_train, y_test, features):
        """
        Selects a feature with the smallest difference in Gini coefficients between train and test sets.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training data features.
        X_test : pd.DataFrame
            Test data features.
        y_train : pd.Series
            Training data target.
        y_test : pd.Series
            Test data target.
        features : list
            List of features in the cluster.

        Returns:
        --------
        str
            Feature with the smallest difference in Gini coefficients between train and test sets.
        """

        gini_diff_scores = {}
        for feature in features:
            gini_train = self._calculate_gini(X_train, y_train, feature)
            gini_test = self._calculate_gini(X_test, y_test, feature)
            gini_diff_scores[feature] = abs(gini_train - gini_test)
        return min(gini_diff_scores, key=gini_diff_scores.get)

    
    def _select_center_feature(self, features, method='kmeans-centroid'):
        """
        Selects the feature closest to the center of the cluster using k-means clustering. 
        
        OR
        
        
        Selects the feature closest to the center of the cluster. The method returns the feature that has 
        the smallest average correlation with all other features in the cluster. 
        This feature is considered the "center" of the cluster because it is, 
        on average, closest (most similar) to the other features in the cluster in terms of correlation.

        *Example*
        Suppose we have a cluster with three features: A, B, and C. 
        The correlation matrix shows the following correlations:

        A B C
        A 1.0 0.8 0.5
        B 0.8 1.0 0.4
        C 0.5 0.4 1.0
        
        For feature A: Average correlation = mean(|0.8|, |0.5|) = 0.65
        For feature B: Average correlation = mean(|0.8|, |0.4|) = 0.6
        For feature C: Average correlation = mean(|0.5|, |0.4|) = 0.45
        Here, feature C has the lowest average correlation with the other features, 
        so C would be selected as the center feature of this cluster.


        Parameters:
        -----------
        features : list
            List of features in the cluster.
            
        method: str
           'kmeans-centroid' - by default, the feature closest to the center of the cluster using k-means clustering
           'min-avg' - the feature that has the smallest average correlation with all other features in the cluster

        Returns:
        --------
        str
            Feature closest to the center of the cluster.
        """
        if method=='kmeans-centroid':
                    
            dist_matrix = 1 - np.abs(self.correlation_matrix.loc[features, features])
            dist_matrix_np = dist_matrix.to_numpy()

            kmeans = KMeans(n_clusters=1, random_state=42)
            kmeans.fit(dist_matrix_np)
            centroid = kmeans.cluster_centers_[0]
            closest_feature = None
            min_distance = float('inf')
            for i, feature in enumerate(features):
                distance = np.linalg.norm(dist_matrix_np[i] - centroid)
                if distance < min_distance:
                    closest_feature = feature
                    min_distance = distance
            return closest_feature
        else:
    
            avg_correlation = {}
            for feature in features:
                correlations = [abs(self.correlation_matrix.loc[feature, f]) for f in features if f != feature]
                avg_correlation[feature] = np.mean(correlations)
            return min(avg_correlation, key=avg_correlation.get)

                

    def fit_transform(self, df, target, input_cols, selection_type, verbose=False):
        """
        Fits the model to the data and transforms the input DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing the data.
        input_cols : list
            List of input feature columns.
        selection_type : str
            Method to use for feature selection ('max_train', 'max_test', 'max_valid', 
                                                 'closest_train_test', 'center_cluster').
                                                 
            'max_train' - feature will be selected by max Gini in cluster on *train* df
            'max_test' - feature will be selected by max Gini in cluster on *test* df
            'max_valid' - feature will be selected by max Gini in cluster on *valid* df
            'closest_train_test' - feature will be selected by minimum Gini difference between *train* and *test* df
            'center_cluster' - Selects the feature closest to the center of the cluster using k-means clustering. 
            
             OR
                               feature will be selected as the center of cluster. Selects the feature closest  \ 
                               to the center of the cluster. The method returns the feature that has the smallest \ 
                               average correlation with all other features in the cluster. This feature is considered \ 
                               the "center" of the cluster because it is,  on average, closest (most similar) to the \
                               other features in the cluster in terms of correlation.
                               

        verbose : bool
            description if needed

        Returns:
        --------
        pd.DataFrame
            DataFrame with selected features.
        """
        if verbose:
            print('Number of features were passed: ', len(input_cols))

        start_time = time.time()
        self.correlation_matrix = self._compute_correlation_matrix(df[input_cols])
        self.correlation_matrix = self.correlation_matrix.fillna(0) # if needed
        
        
        cluster_labels = self._hierarchical_clustering(self.correlation_matrix)
        
        df1 = df[input_cols + ['sample_type', target]]
        X_train = df1[df1['sample_type'] == 0].drop([target, 'sample_type'], axis=1)
        y_train = df1[df1['sample_type'] == 0][target]
        X_test = df1[df1['sample_type'] == 1].drop([target, 'sample_type'], axis=1)
        y_test = df1[df1['sample_type'] == 1][target]
        X_valid = df1[df1['sample_type'] == 2].drop([target, 'sample_type'], axis=1)
        y_valid = df1[df1['sample_type'] == 2][target]

        self.clusters = {}
        selected_features = []
        for cluster_id in np.unique(cluster_labels):
            cluster_features = [input_cols[i] for i in range(len(input_cols)) if cluster_labels[i] == cluster_id]
            selected_feature = self._select_feature(X_train, X_test, 
                                                    X_valid, y_train, 
                                                    y_test, y_valid, 
                                                    cluster_features, selection_type)
            selected_features.append(selected_feature)
            self.clusters[f'cluster_{cluster_id}'] = [selected_feature, cluster_features]
            self.selected_features = selected_features
        end_time = time.time()
        if verbose:
            print('Number of features selected: ', len(selected_features))
            print(f"Clusterization finished in: {round(end_time - start_time,2)} seconds")
        
        return df[selected_features]

    def get_clusters(self):
        """
        Returns the clusters formed after fitting the model.

        Returns:
        --------
        dict
            Dictionary of clusters and their selected features.
        """

        if self.clusters is None:
            raise ValueError("Clusters have not been formed. Call transform method first.")
        return self.clusters
    
    def plot_dendrogram(self, savefig=True, output_name='output.png'):
        """
        Plots a dendrogram of the features and highlights the selected features.
        
        Parameters:
        -----------
        savefig:  bool
            The flag if saving of the dendrogramm plot is needed.
        output_name : str
            Name of the output file with format .png


        Raises:
        -------
        ValueError
            If clusters have not been formed or linkage matrix is not available.
        """

        if self.clusters is None or not hasattr(self, 'linkage_matrix'):
            raise ValueError("Clusters have not been formed or linkage matrix is not available. \
                              Call fit_transform method first.")
        
        selected_features = [v[0] for v in self.clusters.values()]

        def leaf_label_func(label):
            if list(self.correlation_matrix.columns)[label] in selected_features:
                return f'*{list(self.correlation_matrix.columns)[label]}*' # Add markers to highlight selected 
            else:
                return list(self.correlation_matrix.columns)[label]

        plt.figure(figsize=(15, 10))
        dendro = dendrogram(self.linkage_matrix, 
                            labels=list(self.correlation_matrix.columns),
                            orientation = 'left', 
                            leaf_rotation=0, 
                            leaf_font_size=7)
                   
        ax = plt.gca()
        x_labels = ax.get_ymajorticklabels()

        for label in x_labels:
            if label.get_text() in selected_features:
                label.set_color('red')
                label.set_fontweight('bold')

        
        plt.title("Dendrogram of Features")
        plt.xlabel("Feature")
        plt.ylabel("Distance")
        if savefig:
            plt.savefig(output_name, dpi=300)
            
        plt.show()

# Example usage
# df = pd.read_csv('your_data.csv')  # load df
# input_cols = ['list_of_your_features']
# cluster_analysis = ClusterAnalysis(correlation_threshold=0.6)
# selected_df = cluster_analysis.fit_transform(df, input_cols, selection_type='max_train') #docstring for more methods
# clusters = cluster_analysis.get_clusters()
