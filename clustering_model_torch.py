import torch
from torch import nn, optim
from sklearn import metrics
import umap
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
# from sklearn.utils.linear_assignment_ import linear_assignment

import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, dims, act='relu'):
        super(Autoencoder, self).__init__()
        self.dims = dims
        self.act = act

        # Activation function
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError('Invalid activation function.')
        # Encoder layers
        self.encoder = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.encoder.append(nn.Linear(dims[i], dims[i + 1]))
        # Decoder layers
        self.decoder = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.decoder.append(nn.Linear(dims[i], dims[i - 1]))
        # Final decoder layer
        self.final_decoder = nn.Linear(dims[1], dims[0])

    def forward(self, x):
        h = x
        for i in range(len(self.dims) - 1):
            h = self.act(self.encoder[i](h))
        for i in range(len(self.dims) - 2, 0, -1):
            h = self.act(self.decoder[i](h))
        return self.final_decoder(h)

def autoencoder_pretraining(train_dataset, val_dataset, test_dataset):
    dims = [input_dim, 500, 500, 2000, cluster_count]
    model = Autoencoder(dims)
    learning_rate = 1e-3
    num_epochs = 1000
    batch_size = 256
    criterion = nn.MSELoss()  # target is the input itself
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_val_loss = float('inf')
    weights_file = 'best_weights.pth'
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()  # switch to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, data).item()
        val_loss /= len(val_loader)  # average loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_file)
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')


    model.eval()
    test_loss = 0
    model.load_state_dict(torch.load(weights_file))# Load the best weights
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, data).item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

def eval_other_methods(x, y, names=None):
    gmm = mixture.GaussianMixture(covariance_type='full', n_components= clusters_count, random_state=0)
    gmm.fit(x)
    y_pred_prob = gmm.predict_proba(x)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(" | GMM clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    y_pred = KMeans(n_clusters=clusters_count,random_state=0).fit_predict(x)
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(" | K-Means clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    sc = SpectralClustering(n_clusters= clusters_count , random_state=0, affinity='nearest_neighbors')
    y_pred = sc.fit_predict(x)
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print("Spectral Clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    md = float("0.00")
    hle = umap.UMAP(random_state=0, metric='euclidean', n_components=clusters_count, n_neighbors=20, min_dist=md)\
        .fit_transform(x)

    gmm = mixture.GaussianMixture(
        covariance_type='full',
        n_components=clusters_count,
        random_state=0)

    gmm.fit(hle)
    y_pred_prob = gmm.predict_proba(hle)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_accuracy(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

if __name__ == "__main__":
    x =5

