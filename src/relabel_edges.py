#!/usr/bin/env python3
# adapted from https://github.com/BorgwardtLab/filtration_curves

from tqdm import tqdm

import igraph as ig
import networkx as nx

from GraphRicciCurvature.OllivierRicci import OllivierRicci


def relabel_edges_with_curvature(graphs):
    """relables edges with O.F. curvature. Note this requires
    converting to networkx and back. Currently this is simple since we
    have no node or edge labels. But would need to reconsider if we
    extend this to labeled graphs."""
    #
    # save graph label and then convert to networkx (nx loses label)
    y = [graph["label"] for graph in graphs]
    try:
        node_labels = [g.vs["label"] for g in graphs]
    except:
        print("no node labels")
    
    graphs = [ig_to_nx(graph) for graph in tqdm(graphs)]

    # compute curvature and append as edge weight
    for i in tqdm(range(len(graphs))):
        if len(graphs[i].edges()) > 0:  # compute curvature only for graphs with at least one edge
            graphs[i] = compute_curvature(graphs[i])  # replace the current graph with the one returned by compute_curvature()

    # convert back to igraph
    graphs = [ig.Graph.from_networkx(graph) for graph in tqdm(graphs)]

    # add graph label and node labels back
    for idx, label in enumerate(y):
        graphs[idx]["label"] = label
        try:
            graphs[idx].vs["label"] = node_labels[idx]
        except:
            print("no node labels")
    return graphs


def compute_curvature(graph):
    """compute curavture and relabel edges in the graph"""

    orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    for u, v, e in graph.edges(data=True):
        # TODO schauen ob das so stimmt
        graph[u][v]["attribute"] = orc.G[u][v]["ricciCurvature"]

    return graph


def ig_to_nx(graph):
    # First, initialize the NetworkX graph
    G = nx.Graph()

    # Transfer nodes
    for node in graph.vs:
        G.add_node(node.index, **node.attributes())
        
    # Transfer edges
    for edge in graph.es:
        G.add_edge(graph.vs[edge.source].index, graph.vs[edge.target].index, **edge.attributes())

    return G


def relabel_nodes(graphs):
    for idx, graph in enumerate(graphs):
        graph.vs["label"] = [v.degree() for v in graph.vs]
    return graphs
