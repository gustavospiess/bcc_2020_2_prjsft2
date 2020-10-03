import typing as tp
import dataclasses
import networkx as nx
import random
from pyclustering.cluster.kmedoids import kmedoids
import itertools
import functools
import numpy as np

import matplotlib.pyplot as plt




frozen_dataclass = dataclasses.dataclass(frozen=True)


random.seed(a='lorem ip')

@frozen_dataclass
class Input:
    """
    n: int
        Number of vertices 
        n > 0

    max_wth: int
        Maximum number of edges within a community by vertex 
        max_wth > 0

    max_btw: int
        Maximum number of edges between a community by vertex 
        max_btw <= max_wth

    mte: int
        Minimum number of edges in resulting graph

    a: tp.Tuple[float, ...]
        A set of attributes descriptors, i.e., standard deviation.

    k: int
        Number of communities
        k > 0

    teta: float
        Threshold for community attributes homogeneity
        0 <= teta <= 1

    nb_rep: int
        Maximum number of community representatives
        nb_rep > 0
    """
    n: int
    max_wth: int
    max_btw: int
    mte: int
    a: tp.Tuple[float, ...]
    k: int
    teta: float
    nb_rep: int


@dataclasses.dataclass
class AttributedVertex:
    """
    Hashable dataclass to represent an attributed node
    """
    idx: int
    att: tp.Tuple[float, ...]
    com: 'Community' = None
    
    def __hash__(self):
        return hash(self.idx)


@dataclasses.dataclass
class Community:
    """
    Hashable dataclass to represent an community
    """
    idx: int
    representative: 'AttributedVertex'
    population: tp.Set[AttributedVertex]
    graph: nx.Graph

    @property
    def subgraph(self) -> nx.Graph:
        return self.graph.subgraph(self.population)
    
    def __hash__(self):
        return hash(self.idx)


Partition = tp.Set['Community']

class ANCGGenerator(object):
    """
    Actual algorithm implementation.
    Executed when initialized.
    """
    def __init__(self, _input: 'Input'):
        self.__input = _input
        self.__graph = None
        self.__partition = None

        self._generate()

    @property
    def graph(self) -> nx.Graph:
        if self.__graph is None:
            raise Exception('graph not generated')
        return self.__graph

    @property
    def partition(self) -> Partition:
        if self.__partition is None:
            raise Exception('graph not generated')
        return self.__partition

    def _generate(self) -> None:
        """
        Executes, step by step, the generation process
        """
        self._initialize_graph()
        self._initialize_community()
        self._batch_vertex_insertion()
        self._final_edge_insertion()

    def _initialize_graph(self) -> None:
        """
        Initialize the graph with the nodes
        """
        self.__graph = nx.Graph()

        def norm(att: float) -> float:
            return random.normalvariate(0, att)

        for i in range(self.__input.n):
            vertex = AttributedVertex(i, tuple(norm(att) for att in self.__input.a))
            self.__graph.add_node(vertex)

    def _initialize_community(self):
        """
        Initialize the communities.

        It is done by electing random representatives, clustering them by
        distance, limiting the size by the minimum cluster size, choosing the
        combination with minimum distance in each cluster.

        This communities are then iterated to add links between some of this
        nodes, and those are elected as representatives of the Community.
        """
        v_init = random.sample(self.__graph.nodes, self.__input.k *
                self.__input.nb_rep)

        kmedoids_instance = kmedoids(tuple(v.att for v in v_init), range(self.__input.k))
        kmedoids_instance.process()

        clusters = kmedoids_instance.get_clusters()
        min_rep = min(len(c) for c in clusters)

        self.__partition = set()
        for i, c in enumerate(clusters):
            com = Community(i, None, set(v_init[j] for j in c), self.graph)
            for v in com.population:
                v.com = com
            self.__partition.add(com)

            center = gravity_center(tuple(v.att for v in com.population))
            combinations = itertools.combinations(com.population, min_rep)

            def distance_to_center(population):
                return sum(distance(center, v.att) for v in population)

            com.representative = set(min(combinations, key=distance_to_center))
            for v in com.population:
                nodes_in_population = com.population - {v} - set(self.graph.neighbors(v))

                if len(nodes_in_population) == 0:
                    continue;

                max_wth = min(len(nodes_in_population), self.__input.max_wth)

                ewth = random.randint(1, max_wth)
                sp = random.sample(nodes_in_population, ewth)
                for other_v in sp:
                    self.__graph.add_edge(v, other_v)
            com.representative = set(com.population)


    def _batch_vertex_insertion(self):
        """
        Set random elected nodes into community with the closer distance to a
        representative, then for each node being added, generates links withing
        and between communities, then re elect random represents
        """
        to_add = {v for v in self.__graph.nodes if v.com is None}
        while len(to_add) > 0:
            sp = random.sample(to_add, random.randint(1, len(to_add)))
            for node_to_add in sp:
                com = None
                if random.random() < self.__input.teta:
                    com = random.choice(tuple(self.__partition))
                else:
                    def avg_distance(com):
                        distance_gen = (distance(node_to_add.att, r.att) for r in com.representative)
                        return sum(distance_gen)/len(com.representative)
                    com = min(self.__partition, key=avg_distance)
                com.population.add(node_to_add)
                node_to_add.com = com
                to_add.remove(node_to_add)
                self._batch_edge_insertion(node_to_add)
            for com in self.partition:
                com.representative = set(random.sample(com.population,
                    k=min(len(com.population), self.__input.nb_rep)))

    def _batch_edge_insertion(self, node_to_add):
        """
        Adds links to the argument node_to_add
        """
        e_wht = rand_pl(min(len(node_to_add.com.population)-1, self.__input.max_wth))
        com = node_to_add.com
        while com.subgraph.degree(node_to_add) < e_wht:
            new_node = self.rand_edge_wth(node_to_add)
            self.__graph.add_edge(node_to_add, new_node)

        e_btw = rand_pl(min((e_wht, self.__input.max_btw))+1)-1

        while self.graph.degree(node_to_add) - com.subgraph.degree(node_to_add) < e_btw:
            new_node = self.rand_edge_btw(node_to_add)
            self.__graph.add_edge(node_to_add, new_node)

    def _final_edge_insertion(self):
        """
        Add edges linking nodes from the same community that share a neighbor
        until the minimum quantity is reached.
        """
        mte = min(self.__input.mte, sum(len(c.population)*(len(c.population)-1)/2 for c in self.partition))

        while len(self.graph.edges) < mte:
            v = random.sample(self.graph.nodes, k=1)[0]
            neig_list = list(v.com.subgraph.neighbors(v))
            random.shuffle(neig_list)

            for v_a, v_b in itertools.combinations(neig_list, 2):
                if not self.graph.has_edge(v_a, v_b):
                    self.graph.add_edge(v_a, v_b)
                    break

    def rand_edge_wth(self, v: 'AttributedVertex'):
        """
        returns random node in v's community not already connected to v, weighed by its degree
        """
        def degree_wth(u):
            return v.com.subgraph.degree(u)
        possible = tuple(v.com.population - {v} - {self.graph.neighbors(v)})
        weigh = tuple(degree_wth(u) for u in possible)
        return random.choices(possible, weigh)[0]

    def rand_edge_btw(self, v: 'AttributedVertex'):
        """
        return random representative of other community, weighed by the distance to v
        """
        possible = tuple(itertools.chain(*(iter(com.representative) for com in self.partition - {v.com})))
        weigh = tuple(distance(v.att, u.att) ** -1 for u in possible)
        return random.choices(possible, weigh)[0]

    def plt(self, pos: bool = True):
        """
        plots the generated graph with or without positioning the nodes by its attributes
        """
        color_iter = itertools.cycle(['red', 'blue', 'green', 'yellow',
            'purple', 'orange', 'gray', 'darkblue', 'darkgreen', 'pink'])
        colors = {com: col for com, col in zip(self.partition, color_iter)}
        coloring = [colors[v.com] for v in self.graph]
        if (pos):
            positioning = {x: x.att[:2] for x in self.graph}
            nx.draw(self.graph, pos=positioning, node_color=coloring)
        else:
            nx.draw(self.graph, node_color=coloring)
        plt.show()


def gravity_center(cluster: tp.Tuple[tp.Tuple[float, ...]]) -> tp.Tuple[float, ...]:
    """
    Calculates the center of mass a cluster, summing and dividing by the length.
    """

    sums = (sum(att) for att in zip(*cluster))
    div = (att/len(cluster) for att in sums)
    return tuple(div)

def distance(a: tp.Tuple[float, ...], b: tp.Tuple[float, ...]) -> float:
    """
    Calculates the euclidean distance between two vertices
    """
    return np.linalg.norm(np.matrix(a)-b)

def rand_pl(m: int) -> int:
    """
    Random integer distributed by a power law in the limit of the parameter m

    E.g.:

    With m = 2
    returns 1 80% of the time
    returns 2 20% of the time

    With m = 3
    returns 1 73.47% of the time
    returns 2 18.37% of the time
    returns 3  8.16% of the time
    """
    weight = (i**-2 for i in range(1, m+1))
    chs = random.choices(range(1, m+1), tuple(weight))
    return chs[0]

default_input = Input(
        n = 40,
        max_wth = 10,
        max_btw = 1,
        mte = 140,
        a = (10.0, 1.0),
        k = 3,
        teta = 0.00,
        nb_rep = 3
    )

generated = ANCGGenerator(default_input)
generated.plt()
generated.plt(pos=False)

