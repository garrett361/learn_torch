import torch
import torch._dynamo
import torch.nn as nn
import utils
from torch.fx.experimental.proxy_tensor import make_fx

D_MODEL = 8


class TestGraphs:
    def test_get_graph(self) -> None:
        graph_module = torch.fx.symbolic_trace(nn.Linear(D_MODEL, D_MODEL))
        graph_module.print_readable()
        isinstance(graph_module, torch.fx.graph_module.GraphModule)
        print(graph_module.graph)
        assert isinstance(graph_module.graph, torch.fx.graph.Graph)

    def test_get_nodes(self) -> None:
        graph_module = torch.fx.symbolic_trace(nn.Linear(D_MODEL, D_MODEL))
        nodes = [n for n in graph_module.graph.nodes]
        print(nodes)
        for n in nodes:
            assert isinstance(n, torch.fx.node.Node)

    def test_print_gm_backend(self) -> None:
        model = nn.Linear(D_MODEL, D_MODEL)
        inputs = torch.randn(1, D_MODEL)

        torch._dynamo.reset()
        backend = utils.PrintGMBackend()
        fn = torch.compile(backend=backend, dynamic=True)(model)

        # triggers compilation of forward graph on the first run
        out = fn(inputs)
        print(out)

    def test_print_gm_backend_with_break(self) -> None:
        class BreakModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin_0 = nn.Linear(D_MODEL, D_MODEL)
                self.lin_1 = nn.Linear(D_MODEL, D_MODEL)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                outputs = self.lin_0(inputs)
                print(outputs)
                outputs = self.lin_1(outputs)
                return outputs

        model = BreakModule()
        inputs = torch.randn(1, D_MODEL)

        torch._dynamo.reset()
        backend = utils.PrintGMBackend()
        fn = torch.compile(backend=backend, dynamic=True)(model)

        out = fn(inputs)
        print(out)

    def test_print_gm_backend_with_bwd(self) -> None:
        model = nn.Linear(D_MODEL, D_MODEL)
        inputs = torch.randn(1, D_MODEL)

        torch._dynamo.reset()
        backend = utils.PrintGMBackend()
        fn = torch.compile(backend=backend, dynamic=True)(model)

        # triggers compilation of forward graph on the first run
        out = fn(inputs)
        out.pow(2).mean().backward()
        print(out)

    def test_print_gm_backend_with_aot(self) -> None:
        model = nn.Linear(D_MODEL, D_MODEL)
        inputs = torch.randn(1, D_MODEL)

        torch._dynamo.reset()
        backend = utils.PrintGMBackend()
        fn = torch.compile(backend=backend.get_aot_compiler(), dynamic=True)(model)

        # triggers compilation of forward graph on the first run
        out = fn(inputs)
        out.pow(2).mean().backward()
        print(out)

    def test_make_fx(self) -> None:
        t = torch.randn(D_MODEL)
        fn = lambda t: t.cos().cos()
        trace = make_fx(fn, tracing_mode="symbolic")(t)
        trace.print_readable()
        isinstance(trace, torch.fx.graph_module.GraphModule)
