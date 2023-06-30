from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import gudhi
import datetime

from sklearn.preprocessing import StandardScaler


def compute_persistent_hom(
    data, thresh, title="", max_edge_length=10, min_persistence=1.5
):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%d_%m_%H_%M")

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    dgms = ripser(data, maxdim=0, thresh=thresh)["dgms"]
    plot_diagrams(dgms, lifetime=True)
    plt.title(f"{title} - Persistence Diagram")
    plt.savefig(f"persistent_hom_outputs/pers_diag{formatted_time}.png")

    rips_complex = gudhi.RipsComplex(points=data, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=0)

    diag = simplex_tree.persistence(min_persistence=min_persistence)

    print(simplex_tree.betti_numbers())

    gudhi.plot_persistence_barcode(diag)
    plt.title(f"{title} - Persistence Barcode")
    plt.savefig(f"persistent_hom_outputs/pers_barcode{formatted_time}.png")

    plt.show()
