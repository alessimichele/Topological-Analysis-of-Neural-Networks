import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def barplot_grid(neurons_IDs, frequencies, associations, suptitle=""):
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    plt.suptitle(f"Barplot for each cluster, {suptitle}")
    for i in range(10):
        # Determine the position in the subplot grid
        row = i // 5
        col = i % 5

        # Select the appropriate subplot
        ax = axs[row, col]

        threshold = (1 / 2) * (max(frequencies[i]) - min(frequencies[i]))
        # Plotting bar chart
        ax.bar(neurons_IDs[i], frequencies[i])

        # Highlight the bar that reaches or exceeds the threshold
        for j, freq in enumerate(frequencies[i]):
            if freq >= threshold:
                ax.bar(neurons_IDs[i][j], freq, color="red")

        # Adding labels and title
        ax.set_xlabel("Neurons")
        ax.set_ylabel("Frequency of activations")
        ax.set_title(f"Hidden layer neurons activations for cluster {i}")

        # Getting the indices of the five most frequent neurons
        top_indices = frequencies[i].argsort()[-5:][::-1]
        top_neurons = neurons_IDs[i][top_indices]

        # Creating the text to display
        text = (
            "5 best neurons: "
            + ", ".join(map(str, top_neurons))
            + "\n"
            + f"Cluster {i} -> label {associations[i][0]} with prob {round(associations[i][1], 2)}"
        )

        # Adding the text to the plot
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        )

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


def heatmap(neurons_IDs, frequencies, associations, hidden_layer_neurons, suptitle=""):
    # plot heatmap

    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    plt.suptitle(f"Heatmaps for hidden layer neurons activations, {suptitle}")

    mesh_size = int(np.sqrt(hidden_layer_neurons))

    for i in range(10):
        # Create a 10x10 grid of frequencies
        heatmap_data = np.zeros((mesh_size, mesh_size))
        for j, val in zip(neurons_IDs[i], frequencies[i]):
            heatmap_data[int(j / mesh_size), j % mesh_size] = val

        # Determine the position in the subplot grid
        row = i // 5
        col = i % 5

        # Select the appropriate subplot
        ax = axs[row, col]

        # Create the heatmap plot with a gradient color scheme
        im = ax.imshow(heatmap_data, cmap="gray", interpolation="bicubic")

        if hidden_layer_neurons == 100:
            # Add text labels
            for x in range(mesh_size):
                for y in range(mesh_size):
                    num = y * mesh_size + x
                    ax.text(x, y, str(num), color="white", ha="center", va="center")

        ax.set_title(f"Frequency neurons heatmap for cluster {i}")

        # Creating the text to display
        text = f"Cluster {i} -> label {associations[i][0]} with prob {round(associations[i][1], 2)}"

        # Adding the text to the plot
        ax.text(
            0.05,
            -0.1,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        )

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


def connections(
    best_neurons_per_cluster,
    associations,
    hidden_layer_neurons,
    clusters=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    title="",
):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes with positions (0, 1) to (99, 1)
    for x in range(hidden_layer_neurons):
        G.add_node(x, pos=(0, (((2 * 100) / hidden_layer_neurons)) * x))

    # Add nodes with positions (0, 0) to (9, 0)
    for x in range(10):
        G.add_node(x + hidden_layer_neurons, pos=(1, 45 + 3 * x))

    # Extract positions from node attributes
    node_positions = {node: data["pos"] for node, data in G.nodes(data=True)}

    colors = ["r", "b", "g", "c", "m", "y", "k", "w", "orange", "purple"]
    for i, edges in enumerate(best_neurons_per_cluster):
        if i in clusters:
            for edge in edges:
                G.add_edge(hidden_layer_neurons + i, edge, color=colors[i])

    colors = nx.get_edge_attributes(G, "color").values()

    # Draw the graph
    plt.figure(figsize=(5, 15))
    plt.title(
        f"Connections between clusters and 5 neurons most activated by each cluster. {title}"
    )
    nx.draw_networkx_nodes(
        G,
        pos=node_positions,
        node_color="black",
        node_size=30 * (100 / hidden_layer_neurons),
    )

    nx.draw_networkx_edges(G, pos=node_positions, edge_color=colors, width=0.75)

    # Add text to the right of each second set node
    for node, pos in node_positions.items():
        if node >= hidden_layer_neurons:  # Only consider the second set of nodes
            plt.text(
                pos[0] + 0.05,
                pos[1],
                str(
                    f"Cluster{node-hidden_layer_neurons} -> label {associations[node-hidden_layer_neurons][0]}"
                ),
                ha="left",
                va="center",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            )

    # Set axis labels and limits
    plt.ylim(-1, 201)
    plt.xlim(-0.5, 1.5)

    # Show the graph
    plt.tight_layout()
    plt.show()
