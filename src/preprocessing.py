from config import karate_dataset
from scipy.io import mmread
import networkx as nx
import networkx.algorithms.community as nx_comm

a = mmread(str(karate_dataset))
print(type(a))
A = nx.from_scipy_sparse_matrix(a)
print(type(A))
print(A.nodes)
print(A.edges)
print(list(A.neighbors(1)))
print(A.degree)
print(nx.info(A))
score = nx_comm.modularity(A, [{a for a in range(10)}, {a for a in range(10,34)}])
print('score', score)
"""
with open(karate_dataset, 'rb') as f:
    line = f.readline()
    while line:
        print(line)
"""