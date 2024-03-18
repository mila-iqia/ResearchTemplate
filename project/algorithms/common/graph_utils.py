from __future__ import annotations
from typing import Generic, Iterable, TypeVar
import networkx as nx

T = TypeVar("T")


class DiGraph(nx.DiGraph, Generic[T]):
    """Adds a tiny bit of typing information to the `nx.DiGraph` class to make it generic.

    Doesn't change anything *at all* about how this class works.
    """

    nodes: list[T]

    def add_node(self, node_for_adding: T, **attr):
        return super().add_node(node_for_adding, **attr)

    def add_edge(self, u_of_edge: T, v_of_edge: T, **attr):
        return super().add_edge(u_of_edge, v_of_edge, **attr)

    def predecessors(self, n: T) -> Iterable[T]:
        return super().predecessors(n)


def topological_sort(graph: DiGraph[T]) -> list[T]:
    return nx.topological_sort(graph)


def sequential_graph(nodes: list[T]) -> DiGraph[T]:
    """From a basic ordered list of objects, construct a directed graph with the identical ordering
    Particularly useful for constructing a feedforward layered network architecture.

    Inputs
    ------
    node_list:
        list of nodes to connect in order

    Outputs
    -------
    graph:
        nx.DiGraph constructed from the list of nodes
    """
    assert isinstance(nodes, list)
    # make sure that there are only unique objects in the node_list
    assert len(set(nodes)) == len(nodes)
    graph = DiGraph()
    for node_index, node in enumerate(nodes):
        if node_index == 0:
            graph.add_node(node)
        else:
            graph.add_edge(nodes[node_index - 1], node)
    return graph


def multi_stream_graph(stream_list: list[list[T]], reverse=False) -> DiGraph[T]:
    """Constructs N sequential graphs from from a list of individual sequences.

    The last node in each stream must be the same, and all streams are connected at this node
    The graph is structured as stream_1 -> final_node <- stream_2 (the same pattern holds for N>2)
    If reverse is True, the graph is structured as stream_1 <- final_node -> stream_2 (the same
    pattern holds for N > 2)

    Inputs
    ------
    stream_list:
        list of lists containing nodes to connect in order. The final node in each node list must
        be common across all node lists
    reverse:
        Boolean. If true, the graph is constructed then the edge directions are all reversed
    Outputs
    -------
    graph:
        nx.DiGraph constructed from the two streams of nodes
    """
    assert all([isinstance(stream, list) for stream in stream_list])
    # require the final node to be shared by all streams
    assert all([stream[-1] == stream_list[0][-1] for stream in stream_list])
    graph = DiGraph()
    # connect the first stream
    for stream in stream_list:
        for node_index, node in enumerate(stream):
            if node_index == 0:
                graph.add_node(node)
            else:
                graph.add_edge(stream[node_index - 1], node)

    if reverse:
        graph = graph.reverse()

    return graph
