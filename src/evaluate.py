import copy
import random
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import igraph
from igraph import Graph, VertexClustering
from config import karate_dataset, coauthors_dataset

def main2():
    membership = np.load('ans_membership/PSO_ca-coauthors-dblp_0.4107.npy')
    print(membership.shape)
    uni, cnts = np.unique(membership, return_counts=True)
    print(uni.shape)
    print(cnts.sum())
    print(cnts.max())
    
def main():
    #dataset = karate_dataset
    dataset = coauthors_dataset
    print('read mtx file')
    mtx = mmread(str(dataset)).tocsr()
    print('to igraph')
    srcs, tgts = mtx.nonzero()
    graph = Graph(list(zip(srcs.tolist(), tgts.tolist())))
    print('start evaluate')

    membership = np.load('ans_membership/PSO_ca-coauthors-dblp_0.4107.npy')
    score = graph.modularity(membership)
    print(score)

if __name__ == '__main__':
    main2()