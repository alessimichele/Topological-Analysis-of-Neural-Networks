from sklearn.decomposition import KernelPCA
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


def plot_manifold(
    data,
    predicted,
    title=["kmeans", "umap", "tsne", "mapper"],
    i=None,
    n_components=3,
    algorithm="umap",
    save_as="",
):
    # i is the i-th cluster for which we want to project
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

    plt.title(
        f"3d projections of hidden activations for test point in cluster {i}, {title}"
    )
    plt.tight_layout()
    plt.savefig(
        f"plot_outputs/{title}/manifold_hidden_activ_per_cluster/cluster_{i}_{save_as}.png"
    )
    plt.show()


from sklearn.decomposition import KernelPCA
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


def plot_manifold_running(
    data, predicted, save_as, title="", n_components=3, algorithm="umap"
):
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
    plt.savefig(f"plot_outputs/{title}_{save_as}.png")
    plt.show()
