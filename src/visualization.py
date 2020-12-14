import graphistry
from config import karate_dataset, coauthors_dataset, ans_dir
from scipy.io import mmread
from scipy.sparse import csr_matrix
import igraph
from igraph import Graph, VertexClustering
import pandas
import numpy as np
from pathlib import Path

#graphistry.register(api=3, protocol="https", server="hub.graphistry.com", token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6InJheXJheXJheTU4NzI0IiwiaWF0IjoxNjA3OTI4MzUyLCJleHAiOjE2MDc5MzE5NTIsInVzZXJfaWQiOjIwNzksIm9yaWdfaWF0IjoxNjA3OTI4MzUyfQ.CfdWBwtCCrjjoddMdsRYVM-6GBsZEipKy7bA-FW3p5M")
#graphistry.register(api=3, username='rayrayray58724', password='graphistry')
#graphistry.register(api=3, protocol="https", server="hub.graphistry.com", token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6InJheXJheXJheTU4NzI0IiwiaWF0IjoxNjA3OTMyODg3LCJleHAiOjE2MDc5MzY0ODcsInVzZXJfaWQiOjIwNzksIm9yaWdfaWF0IjoxNjA3OTMyODg3fQ.EXidac1idPt0FhFY0GWCM-zRBXNGQG8aMKhqaGyyeeM")

#dataset = karate_dataset
dataset = coauthors_dataset
print('read mtx file')
mtx = mmread(str(dataset)).tocsr()
print('to igraph')
srcs, tgts = mtx.nonzero()
graph = Graph(
    list(zip(srcs.tolist(), tgts.tolist())), 
    edge_attrs={"weight":np.ones(srcs.shape[0])},
    vertex_attrs={"radius":np.ones(max(srcs.max(), tgts.max())+1).astype(int)}
)
#print(list(graph.vs['radius']))
print('start')
#membership = np.load(ans_dir/Path('ACO_soc-karate_0.3944.npy'))
membership = np.load(ans_dir/Path('PSO_ca-coauthors-dblp_0.3998.npy'))
print(membership)

### fast test with subgraph
graph = graph.subgraph(list(range(50000)))
#membership = membership[:50000]
membership = np.random.randint(50000, size=(50000))

print('contracting')
graph.contract_vertices(membership, combine_attrs="sum")
graph.simplify(multiple=True, loops=False, combine_edges="sum")
#print(list(graph.es))
print(graph.vs["radius"])

print('plotting')
print(graph.vcount(), graph.ecount())
super_edge_weight = np.array(graph.es["weight"])
super_node_size = np.array(graph.vs["radius"])
#norm_super_edge_weight = super_edge_weight / super_edge_weight.sum()
norm_super_edge_weight = (super_edge_weight - super_edge_weight.min()) / (super_edge_weight.max() - super_edge_weight.min())
norm_super_node_size = (super_node_size - super_node_size.min()) / (super_node_size.max() - super_node_size.min())

#print(norm_super_edge_weight)
#visual_style = {}
#widths = [e["weight"] for e in graph.es]
igraph.plot(graph,'out.png',
    edge_width = norm_super_edge_weight*20,
    vertex_size = norm_super_node_size* 20,
    layout=graph.layout('circle')
)