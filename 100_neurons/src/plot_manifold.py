from sklearn.decomposition import KernelPCA
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


def plot_manifold(data, predicted, title="", n_components=3, algorithm="umap"):
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    if algorithm == "umap":
        reducer = umap.UMAP(n_components=n_components)
    elif algorithm == "KernelPCA":
        reducer = KernelPCA(n_components=n_components)
    else:
        raise ValueError("Enter one of:[umap, KernelPCA]")

    output_data = reducer.fit_transform(data_normalized)

    if n_components == 2:
        plt.scatter(output_data[:, 0], output_data[:, 1], c=predicted)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the data points
        ax.scatter(output_data[:, 0], output_data[:, 1], output_data[:, 2], c=predicted)

    plt.title(title)
    plt.show()
