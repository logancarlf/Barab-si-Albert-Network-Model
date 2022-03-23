import networkx as net
import numpy as np


class BA_Model:

    def phase_1_preferential_attachment(self, n, m):
        self.__network = net.star_graph(m)
        repeated_nodes = [n for n, d in self.__network.degree() for _ in range(d)]
        source = len(self.__network)
        while source < n:
            r = np.random.randint(0, len(repeated_nodes)-1, size=m)
            addedge = [repeated_nodes[i] for i in r]
            self.__network.add_edges_from(zip([source] * m, addedge))
            repeated_nodes.extend(addedge)
            repeated_nodes.extend([source] * m)

            source += 1
        return self.__network
