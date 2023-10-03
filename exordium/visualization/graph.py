from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx


class GraphVisualization:

    def __init__(self):
        self.G = None
        self.visual = []

    def add_edge(self, a, b):
        self.visual.append([a, b])

    def visualize(self, ax: Axes = None, output: str | Path = None):
        self.G = nx.Graph()
        self.G.add_edges_from(self.visual)

        if ax is None:
            _, ax = plt.subplots()

        nx.draw(self.G, with_labels=True,
                        node_size=1,
                        font_size=5,
                        node_color="skyblue",
                        edge_color="grey",
                        node_shape=".",
                        alpha=0.8,
                        linewidths=10)

        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output)
            plt.close()

    def number_of_nodes(self):
        return self.G.number_of_nodes()

    def number_of_edges(self):
        return self.G.number_of_edges()