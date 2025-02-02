import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Inizializza il modello PCA.
        
        Args:
            n_components (int): Numero di componenti principali da mantenere. 
                                Se None, mantiene tutte le dimensioni.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Calcola i componenti principali e la varianza spiegata.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        """
        # Standardize data
        self.mean_ = np.mean(X, axis=0)
        X = (X - self.mean_) / np.std(X, axis=0)

        # Compute covariance matrix
        cov = np.cov(X.T)

        # Compute eigenvalues and eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[sorted_indices] # vector of eigenvalues in descending order
        eigenvectors = eigenvectors[:, sorted_indices] # matrix of eigenvectors in descending order

        # Normalize eigenvectors
        eigenvectors /= np.linalg.norm(eigenvectors, axis=0)

        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance

        # Keep only the top n_components eigenvectors
        if self.n_components is not None:
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]

        # Store components and explained variance
        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues

    def transform(self, X):
        """
        Proietta i dati sullo spazio dei componenti principali.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        
        Returns:
            numpy.ndarray: Dati proiettati nello spazio ridotto di forma (n_samples, n_components).
        """
        # Standardize data
        X = (X - self.mean_) / np.std(X, axis=0)

        # Project data onto principal components
        return np.dot(X, self.components_)

    def fit_transform(self, X):
        """
        Fit the model and transform the data.
        
        Args:
            X (numpy.ndarray): Dataset di input di forma (n_samples, n_features).
        
        Returns:
            numpy.ndarray: Dati proiettati nello spazio ridotto di forma (n_samples, n_components).
        """
        # Fit the model and transform the data
        self.fit(X)
        return self.transform(X)
    
    def get_explained_variance_ratio(self):
        """
        Restituisce la varianza spiegata per ogni componente principale.
        
        Returns:
            numpy.ndarray: Vettore di varianza spiegata per ogni componente principale.
        """
        return self.explained_variance_ratio_