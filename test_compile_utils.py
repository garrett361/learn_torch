import torch
import torch._dynamo
import torch.nn as nn


def toy_backend(gm, sample_inputs):
    print("Dynamo produced a fx Graph in Torch IR:")
    gm.print_readable()

    print("Notice that sample_inputs is a list of flattened FakeTensor:")
    print(sample_inputs)
    return gm.forward


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

    def test_toy_backend(self) -> None:
        # https://colab.research.google.com/drive/1Zh-Uo3TcTH8yYJF-LLo5rjlHVMtqvMdf?usp=sharing#scrollTo=UklMVs56u9j7
        model = nn.Linear(8, 8)
        inputs = torch.randn(1, 8)

        torch._dynamo.reset()
        fn = torch.compile(backend=toy_backend, dynamic=True)(model)

        # triggers compilation of forward graph on the first run
        out = fn(inputs)
        print(out)
