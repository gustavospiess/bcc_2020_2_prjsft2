import typing as tp
import dataclasses
import networkx as nx
import random as rand
from pyclustering.cluster.kmedoids import kmedoids
import itertools
import numpy as np

import matplotlib.pyplot as plt


frozenDataClass = dataclasses.dataclass(frozen=True)

@frozenDataClass
class Input:
    """ n: int
        Number of vertices 
        n > 0

    max_wth: int
        Maximum number of edges within a community by vertex 
        max_wth > 0

    max_btn: int
        Maximum number of edges between a community by vertex 
        max_btn <= max_wth

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

    nbRep: int
        Maximum number of community representatives
        nbRep > 0
    """
    n: int
    max_wth: int
    max_btn: int
    mte: int
    a: tp.Tuple[float, ...]
    k: int
    teta: float
    nbRep: int


@dataclasses.dataclass
class AttributedVertex:
    idx: int
    att: tp.Tuple[float, ...]
    com: 'Community' = None
    
    def __hash__(self):
        return hash(self.idx)


@dataclasses.dataclass
class Community:
    idx: int
    representant: 'AttributedVertex'
    population: tp.Set[AttributedVertex]
    
    def __hash__(self):
        return hash(self.idx)


Partition = tp.Set['Community']

class Generator(object):
    def __init__(self, _input: 'Input'):
        self.__input = _input
        self.__graph = None
        self.__partition = None

        self._generate()

    @property
    def graph(self):
        if self.__graph is None:
            raise Exception('ungenerated graph')
        return self.__graph

    @property
    def partition(self):
        if self.__partition is None:
            raise Exception('ungenerated graph')
        return self.__partition

    def _generate(self) -> tp.Tuple[nx.Graph, Partition]:
        self._initialize_graph()
        self._initialize_community()
        self._batch_vertex_insertion()

    def _initialize_graph(self):
        self.__graph = nx.Graph()
        norm = lambda att: rand.normalvariate(0, att)
        for i in range(self.__input.n):
            vertex = AttributedVertex(i, tuple(map(norm, self.__input.a)), None)
            self.__graph.add_node(vertex)

    def _initialize_community(self):
        v_init = rand.sample(self.__graph.nodes, self.__input.k * self.__input.nbRep)

        att = lambda v: v.att

        kmedoids_instance = kmedoids(tuple(map(att, v_init)), range(self.__input.k))
        kmedoids_instance.process()

        clusters = kmedoids_instance.get_clusters()
        minRep = min(map(len, clusters))

        self.__partition = set()
        self.plt()
        for i, c in enumerate(clusters):
            com = Community(i, None, set(map(lambda i: v_init[i], c)))
            for v in com.population:
                v.com = com
            self.__partition.add(com)

            center = gravityCenter(tuple(map(att, com.population)))
            combinations = itertools.combinations(com.population, minRep)
            com.representant = set(min(combinations, key=lambda population: sum(map(lambda v: distance(center, v.att), population))))
            for v in com.population:
                nodes_in_population = com.population - {v} - set(self.graph.neighbors(v))

                if len(nodes_in_population) == 0:
                    continue;

                max_wth = min(len(nodes_in_population), self.__input.max_wth)

                ewth = rand.randint(1, max_wth)
                sp = rand.sample(nodes_in_population, ewth)
                for other_v in sp:
                    self.__graph.add_edge(v, other_v)
            self.plt()


    def _batch_vertex_insertion(self):
        to_add = set(filter(lambda v: v.com is None, self.__graph.nodes))
        while len(to_add) > 0:
            self.plt()
            sp = rand.sample(to_add, rand.randint(1, len(to_add)))
            for node_to_add in sp:
                com = None
                if rand.random() < self.__input.teta:
                    com = rand.choice(tuple(self.__partition))
                else:
                    com = min(self.__partition,
                            key=lambda com: sum(map(lambda r: distance(node_to_add.att, r.att), com.representant))/len(com.representant))
                com.population.add(node_to_add)
                node_to_add.com = com
                to_add.remove(node_to_add)

    def plt(self):
        positioning = {x: x.att for x in self.graph}
        colors = {com: col for com, col in zip(self.partition, ['red', 'blue', 'green'])}
        colors[None] = 'grey'
        coloring = [colors[v.com] for v in self.graph]
        nx.draw(self.graph, pos=positioning, node_color=coloring)
        plt.show()


def gravityCenter(cluster: tp.Iterable[tp.Tuple[float, ...]]) -> tp.Tuple[float, ...]:
    t = len(cluster)
    l = len(cluster[0])
    return tuple([sum(map(lambda p: p[i], cluster))/t for i in range(l)])


def distance(a: tp.Tuple[float, ...], b: tp.Tuple[float, ...]) -> float:
    return np.linalg.norm(np.matrix(a)-b)


default_input =  Input(
        n = 30,
        max_wth = 10,
        max_btn = 2,
        mte = 10,
        a = (10.0, 10.0),
        k = 3,
        teta = 0.1,
        nbRep = 5
    )


generated = Generator(default_input)
generated.plt()

