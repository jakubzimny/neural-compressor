import pytest
import torch

from neural_compressor.torch.export import export
from neural_compressor.torch.quantization import StaticQuantConfig, convert, prepare


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.mark.skipif(not is_xpu_available(), reason="XPU device is not available")
@torch.no_grad()
def test_static_int8_quantization_smoke():
    device = torch.device("xpu")
    float_model = TinyMLP().to(device)
    float_model.eval()
    example_input = torch.randn(1, 4, device=device)
    exported_model = export(float_model, example_inputs=(example_input,))
    assert exported_model is not None
    #reference_output = float_model(example_input)

    quant_config = StaticQuantConfig(
        w_dtype="int8",
        w_granularity="per_tensor",
        act_dtype="uint8",
        act_granularity="per_tensor",
    )

    prepared_model = prepare(exported_model, quant_config, example_inputs=(example_input,))
    #prepared_model(example_input)

    quantized_model = convert(prepared_model)

    graph = getattr(quantized_model, "graph", None)
    if graph is None and hasattr(quantized_model, "graph_module"):
        graph = quantized_model.graph_module.graph
    assert graph is not None, "Quantized model does not expose an FX graph for inspection"

    quantized_nodes = [
        node
        for node in graph.nodes
        if "quantize" in str(node.target) or "dequantize" in str(node.target)
    ]
    assert quantized_nodes, "No quantize/dequantize ops detected; model may not be quantized"

    #quant_output = quantized_model(example_input)

    #assert quant_output.shape == reference_output.shape
    #assert torch.isfinite(quant_output).all(), "Quantized output contains non-finite values"
    assert quant_output.device.type == "xpu", "Quantized model did not stay on XPU device"
