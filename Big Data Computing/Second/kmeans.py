import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, kmeanspp=False, random_state=None):
        """
        Inizializza il modello K-Means.
        
        Args:
            n_clusters (int): Numero di cluster da trovare.
            max_iter (int): Numero massimo di iterazioni.
            tol (float): Tolleranza per determinare la convergenza.
            random_state (int): Seed per la riproducibilità.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kmeanspp = kmeanspp
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
    
    def _find_closest_centroids(self, X):
        """
        Trova i centroidi piü vicini per ogni dato.

        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        
        Returns:
            numpy.ndarray: Indici dei centroidi piü vicini per ogni dato.
        """

        # Calcolo la distanza dei dati dai centroidi
        distances = [np.linalg.norm(X - c, axis=1)**2 for c in self.centroids_]

        # Trovo il centroide più vicino per ogni dato
        return np.argmin(distances, axis=0)
    
    def _kmeanspp_init(self, X):
        """
        Utilizza K-Means++ per inizializzare i centroidi.

        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).

        Returns:
            numpy.ndarray: Centroidi iniziali.
        """
        n_samples, n_features = X.shape

        # Scelgo il primo centroide casualmente
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            # Calcolo la distanza minima al quadrato per ogni punto rispetto ai centroidi già scelti
            distances = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids], axis=0)

            # Calcolo le probabilità proporzionali alle distanze
            probabilities = distances / distances.sum()

            # Scelgo un nuovo centroide in base alle probabilità
            new_centroid = X[np.random.choice(n_samples, p=probabilities)]
            centroids.append(new_centroid)

        return np.array(centroids)
    
    def _compute_inertia(self, X, centroids, labels):
        """
        Calcola l'inerzia (somma delle distanze al quadrato dai punti ai loro centroidi).
        
        Args:
            X (numpy.ndarray): Dataset di input.
            centroids (numpy.ndarray): Centroidi dei cluster.
            labels (numpy.ndarray): Etichette dei cluster.
        
        Returns:
            float: L'inerzia totale.
        """
        return np.sum([np.sum((X[labels == k] - centroids[k]) ** 2) for k in range(self.n_clusters)])
    
    def _compute_silhouette_score(self, X):
        """
        Calcola il punteggio silhouette per i cluster.

        Args:
            X (numpy.ndarray): Dataset di input.

        Returns:
            float: Il punteggio silhouette medio.
        """
        n_samples = len(X)
        a = np.zeros(n_samples)  # Coesione: distanza media intra-cluster
        b = np.zeros(n_samples)  # Separazione: distanza media verso il cluster più vicino
        
        for i in range(n_samples):
            cluster = self.labels_[i]
            cluster_points = X[self.labels_ == cluster]
            
            # Calcolo di 'a': distanza media intra-cluster, escludendo il punto X[i]
            if len(cluster_points) > 1:
                distances = np.linalg.norm(cluster_points - X[i], axis=1)
                a[i] = np.sum(distances) / (len(cluster_points) - 1)
            else:
                a[i] = 0  # Cluster con un solo punto
            
            # Calcolo di 'b': distanza media verso il cluster più vicino
            b[i] = np.min([
                np.mean(np.linalg.norm(X[self.labels_ == k] - X[i], axis=1))
                for k in range(self.n_clusters) if k != cluster
            ])
        
        # Calcolo del punteggio silhouette per ogni punto
        s = (b - a) / np.maximum(a, b)
        s = np.nan_to_num(s)  # Gestione di eventuali NaN (es. quando a == b == 0)

        # Restituisce il punteggio silhouette medio
        return np.mean(s)
    
    def _compute_dbi(self, X):

        def _get_cluster_average_internal_distance(X, cluster):
            cluster_points = X[self.labels_ == cluster]

            return np.sum(np.linalg.norm(cluster_points - self.centroids_[cluster], axis=1)) / (len(cluster_points) - 1)
        dbis = []
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            cluster_average_distance = _get_cluster_average_internal_distance(X, k)

            tmp_dbis = []

            for j in range(self.n_clusters):
                if j != k:
                    other_cluster_average_distance = _get_cluster_average_internal_distance(X, j)

                    distance = np.linalg.norm(self.centroids_[k] - self.centroids_[j])

                    tmp_dbis.append((cluster_average_distance + other_cluster_average_distance) / distance)
            dbis.append(max(tmp_dbis))

        return np.mean(dbis)


    def _compute_dunn_index(self, X):
        """
        Calcola l'indice di Dunn.
        
        Args:
            X (numpy.ndarray): Dataset di input.
        
        Returns:
            float: L'indice di Dunn.
        """
        def _get_cluster_diameter(X, cluster):
            cluster_points = X[self.labels_ == cluster]
            return np.max(np.linalg.norm(cluster_points - self.centroids_[cluster], axis=1))

        # Calcola il diametro di ciascun cluster
        cluster_diameters = [_get_cluster_diameter(X, k) for k in range(self.n_clusters)]

        # Calcola la distanza minima tra i centroidi
        min_intercluster_distance = np.min([
            np.linalg.norm(self.centroids_[i] - self.centroids_[j])
            for i in range(self.n_clusters)
            for j in range(i + 1, self.n_clusters)
        ])

        # Restituisce l'indice di Dunn
        return min_intercluster_distance / max(cluster_diameters)
    
    def fit(self, X):
        """
        Esegue il clustering K-Means sui dati.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        """
        # Imposta il seed per la riproducibilità
        np.random.seed(self.random_state)

        # Inizializza i centroidi

        if self.kmeanspp:
            # Utilizza K-Means++ per inizializzare i centroidi
            self.centroids_ = self._kmeanspp_init(X)
        else:
            # Utilizza una distribuzione uniforme per inizializzare i centroidi
            self.centroids_ = X[np.random.choice(len(X), self.n_clusters, replace=False)]

        # Esegue il clustering K-Means
        for _ in range(self.max_iter):
            # Trova i centroidi più vicini
            labels = self._find_closest_centroids(X)

            # Calcola i nuovi centroidi
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Calcola la variazione dei centroidi
            delta = np.linalg.norm(new_centroids - self.centroids_)

            # Aggiorna i centroidi
            self.centroids_ = new_centroids

            # Se la variazione è minore della tolleranza, esce dal ciclo
            if delta < self.tol:
                break

        # Memorizza i risultati finali
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, self.centroids_, labels)
    
    def predict(self, X):
        """
        Predice il cluster di appartenenza per ogni dato in X.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        
        Returns:
            numpy.ndarray: Indici dei cluster per ogni dato.
        """
        return self._find_closest_centroids(X)
    
    def fit_predict(self, X):
        """
        Esegue il clustering K-Means sui dati e restituisce i cluster di appartenenza.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        
        Returns:
            numpy.ndarray: Indici dei cluster per ogni dato.
        """
        self.fit(X)
        return self.labels_
    
    def plot_clusters(self, X):
        """
        Visualizza i cluster e i centroidi.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        """
        import matplotlib.pyplot as plt

        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, cmap='viridis', s=50, alpha=0.5)
        
        # Plot centroids
        plt.scatter(self.centroids_[:, 0], self.centroids_[:, 1], c='red', s=200, marker='x')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-Means Clustering')
        plt.show()
    
    def plot_inertia(self, X):
        """
        Visualizza l'inerzia in funzione del numero di cluster.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        """
        import matplotlib.pyplot as plt

        inertias = []
        for k in range(1, 11):
            model = KMeans(n_clusters=k, random_state=self.random_state)
            model.fit(X)
            inertias.append(model.inertia_)
        
        plt.plot(range(1, 11), inertias, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('K-Means Inertia')
        plt.show()
    
    def plot_silhouette_score(self, X):
        """
        Visualizza il punteggio silhouette in funzione del numero di cluster.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        """
        import matplotlib.pyplot as plt

        silhouette_scores = []
        for k in range(2, 11):
            model = KMeans(n_clusters=k, random_state=self.random_state)
            model.fit(X)
            silhouette_scores.append(model._compute_silhouette_score(X))

        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('K-Means Silhouette Score')
        plt.show()

    def plot_elbow(self, X):
        """
        Visualizza l'elbow method in funzione del numero di cluster.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        """
        import matplotlib.pyplot as plt

        inertias = []
        for k in range(1, 11):
            model = KMeans(n_clusters=k, random_state=self.random_state)
            model.fit(X)
            inertias.append(model.inertia_)
        
        plt.plot(range(1, 11), inertias, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('K-Means Inertia')
        plt.show()

    def plot_dbi(self, X):
        """
        Visualizza il Davies-Bouldin Index in funzione del numero di cluster.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        """
        import matplotlib.pyplot as plt

        dbis = []
        for k in range(2, 11):
            model = KMeans(n_clusters=k, random_state=self.random_state)
            model.fit(X)
            dbis.append(model._compute_dbi(X))

        plt.plot(range(2, 11), dbis, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Davies-Bouldin Index')
        plt.title('K-Means Davies-Bouldin Index')
        plt.show()
    
    def plot_dunn_index(self, X):
        """
        Visualizza l'indice di Dunn in funzione del numero di cluster.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        """
        import matplotlib.pyplot as plt

        dunn_indices = []
        for k in range(2, 11):
            model = KMeans(n_clusters=k, random_state=self.random_state)
            model.fit(X)
            dunn_indices.append(model._compute_dunn_index(X))

        plt.plot(range(2, 11), dunn_indices, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Dunn Index')
        plt.title('K-Means Dunn Index')
        plt.show()