import torch
import torch.nn as nn


class TestGraphs:
    def test_get_graph(self) -> None:
        graph_module = torch.fx.symbolic_trace(nn.Linear(8, 8))
        graph_module.print_readable()
        print(graph_module.graph)
        assert isinstance(graph_module.graph, torch.fx.graph.Graph)

    def test_get_nodes(self) -> None:
        graph_module = torch.fx.symbolic_trace(nn.Linear(8, 8))
        nodes = [n for n in graph_module.graph.nodes]
        print(nodes)
        for n in nodes:
            assert isinstance(n, torch.fx.node.Node)
