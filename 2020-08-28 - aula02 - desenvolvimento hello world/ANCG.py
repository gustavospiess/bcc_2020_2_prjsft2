import typing as tp
import dataclasses
import networkx as nx
import random as rand
from pyclustering.cluster.kmedoids import kmedoids
import itertools
import numpy as np


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
    rep: 'AttributedVertex'
    pop: tp.Set[AttributedVertex]
    
    def __hash__(self):
        return hash(self.idx)


Partition = tp.Set['Community']


def generate(_input: 'Input') -> tp.Tuple[nx.Graph, Partition]:
    g = nx.Graph()

    norm = lambda att: rand.normalvariate(0, att)
    for i in range(_input.n):
        vertex = AttributedVertex(i, tuple(map(norm, _input.a)), None)
        g.add_node(vertex)

    p = initialize_community(g, _input)

    return (g,p)

def gravityCenter(cluster: tp.Iterable[tp.Tuple[float, ...]]) -> tp.Tuple[float, ...]:
    t = len(cluster)
    l = len(cluster[0])
    return tuple([sum(map(lambda p: p[i], cluster))/t for i in range(l)])

def distance(a: tp.Tuple[float, ...], b: tp.Tuple[float, ...]) -> float:
    return np.linalg.norm(np.matrix(a)-b)

def initialize_community(g: nx.Graph, _input: 'Input') -> Partition:
    v_init = rand.sample(g.nodes, _input.k * _input.nbRep)

    att = lambda v: v.att

    kmedoids_instance = kmedoids(tuple(map(att, v_init)), range(_input.k))
    kmedoids_instance.process()

    clusters = kmedoids_instance.get_clusters()
    minRep = min(map(len, clusters))

    p = set()
    for i, c in enumerate(clusters):
        com = Community(i, None, set(map(lambda i: v_init[i], c)))
        for v in com.pop:
            v.com = com
        p.add(com)

        center = gravityCenter(tuple(map(att, com.pop)))
        combinations = itertools.combinations(com.pop, minRep)
        com.rep = set(min(combinations, key=lambda pop: sum(map(lambda v: distance(center, v.att), pop))))

    return p




default_input =  Input(
        n = 3,
        max_wth = 14,
        max_btn = 4,
        mte = 150,
        a = (1.0, 5.0),
        k = 3,
        teta = 0,
        nbRep = 5
    )


p = generate(default_input)[1]
print(p)
for c in p:
    print(tuple(map(lambda v: v.idx, c.pop)))
    # print(tuple(map(lambda v: v.idx, c.rep)))

