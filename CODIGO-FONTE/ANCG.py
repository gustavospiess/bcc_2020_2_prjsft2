import typing as tp
import dataclasses
import networkx as nx
import random
from pyclustering.cluster.kmedoids import kmedoids
import itertools
import functools
import numpy as np

import matplotlib.pyplot as plt




frozenDataClass = dataclasses.dataclass(frozen=True)


random.seed(a='lorem ip')

@frozenDataClass
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

    nbRep: int
        Maximum number of community representatives
        nbRep > 0
    """
    n: int
    max_wth: int
    max_btw: int
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
    graph: nx.Graph

    @property
    def subgraph(self) -> nx.Graph:
        return self.graph.subgraph(self.population)
    
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
        self._final_edge_insertion()

    def _initialize_graph(self):
        self.__graph = nx.Graph()
        norm = lambda att: random.normalvariate(0, att)
        for i in range(self.__input.n):
            vertex = AttributedVertex(i, tuple(map(norm, self.__input.a)), None)
            self.__graph.add_node(vertex)

    def _initialize_community(self):
        v_init = random.sample(self.__graph.nodes, self.__input.k * self.__input.nbRep)

        att = lambda v: v.att

        kmedoids_instance = kmedoids(tuple(map(att, v_init)), range(self.__input.k))
        kmedoids_instance.process()

        clusters = kmedoids_instance.get_clusters()
        minRep = min(map(len, clusters))

        self.__partition = set()
        # self.plt()
        for i, c in enumerate(clusters):
            com = Community(i, None, set(map(lambda i: v_init[i], c)), self.graph)
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

                ewth = random.randint(1, max_wth)
                sp = random.sample(nodes_in_population, ewth)
                for other_v in sp:
                    self.__graph.add_edge(v, other_v)
                # self.plt()
            com.representant = set(com.population)


    def _batch_vertex_insertion(self):
        to_add = set(filter(lambda v: v.com is None, self.__graph.nodes))
        # self.plt()
        while len(to_add) > 0:
            sp = random.sample(to_add, random.randint(1, len(to_add)))
            for node_to_add in sp:
                com = None
                if random.random() < self.__input.teta:
                    com = random.choice(tuple(self.__partition))
                else:
                    com = min(self.__partition,
                            key=lambda com: sum(map(lambda r: distance(node_to_add.att, r.att), com.representant))/len(com.representant))
                com.population.add(node_to_add)
                node_to_add.com = com
                to_add.remove(node_to_add)
                self._batch_edge_insertion(node_to_add)
            for com in self.partition:
                com.representant = set(random.sample(com.population, k=min(len(com.population), self.__input.nbRep)))
            # self.plt()

    def _batch_edge_insertion(self, node_to_add):
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
        mte = min(self.__input.mte, sum(len(c.population)*(len(c.population)-1)/2 for c in self.partition))

        tested = set()

        while len(tested) < len(self.graph.edges) < mte:
            v = random.sample(self.graph.nodes - tested, k=1)[0]
            neig_list = list(v.com.subgraph.neighbors(v))
            random.shuffle(neig_list)

            pair_iter = filter(lambda p: not self.graph.has_edge(*p), itertools.combinations(neig_list, 2))
            for pair in pair_iter:
                self.graph.add_edge(*pair)
                break
            else:
                tested.add(v)

    def rand_edge_wth(self, v: 'AttributedVertex'):
        degree_wth = v.com.subgraph.degree
        total = max(sum(degree_wth(u) for u in v.com.population), 1)
        possible = tuple(v.com.population - {v} - {self.graph.neighbors(v)})
        weith = tuple(degree_wth(u)/total for u in v.com.population - {v})
        return random.choices(possible, weith)[0]

    def rand_edge_btw(self, v: 'AttributedVertex'):
        possible = tuple(itertools.chain(*(iter(com.representant) for com in self.partition)))
        total = sum(distance(v.att, u.att) ** -1 for u in possible)
        weith = tuple(distance(v.att, u.att) ** -1 / total for u in possible)
        return random.choices(possible, weith)[0]

    def plt(self, pos=True):
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


def gravityCenter(cluster: tp.Iterable[tp.Tuple[float, ...]]) -> tp.Tuple[float, ...]:
    t = len(cluster)
    l = len(cluster[0])
    return tuple([sum(map(lambda p: p[i], cluster))/t for i in range(l)])


# @functools.lru_cache(maxsize=2048)
def distance(a: tp.Tuple[float, ...], b: tp.Tuple[float, ...]) -> float:
    return np.linalg.norm(np.matrix(a)-b)

def rand_pl(m: int) -> int:
    total = sum(i**-2 for i in range(1, m+1))
    return random.choices(range(1, m+1), tuple(i**-2/total for i in range(1, m+1)))[0]

default_input = Input(
        n = 100,
        max_wth = 20,
        max_btw = 5,
        mte = 700,
        a = (1.0, 1.0),
        k = 10,
        teta = 0.01,
        nbRep = 3
    )

generated = Generator(default_input)
generated.plt()
generated.plt(pos=False)

