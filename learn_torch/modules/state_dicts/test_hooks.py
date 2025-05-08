import torch
import torch.nn as nn


class Experts(nn.Module):
    def __init__(self, d_model: int, n_experts: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        for idx in range(n_experts):
            setattr(self, str(idx), nn.Parameter(torch.randn(d_model)))


class ExpertsConsolidated(nn.Module):
    def __init__(self, d_model: int, n_experts: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self._weights = nn.Parameter(torch.randn(n_experts, d_model))
        self.register_state_dict_pre_hook(self.state_dict_pre_hook)
        self.register_state_dict_post_hook(self.state_dict_post_hook)
        self.register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)

    @staticmethod
    def state_dict_pre_hook(module, prefix, keep_vars) -> None:
        _weight_data = module._weights.data
        # De-register consolidated weights
        delattr(module, "_weights")
        module._parameters = {n: p for n, p in module._parameters.items() if n != "_weights"}

        # Keep reference to data around for later
        module._weight_data = _weight_data
        with torch.no_grad():
            for idx, chunk in enumerate(_weight_data):
                module.register_parameter(str(idx), nn.Parameter(chunk))

    @staticmethod
    def state_dict_post_hook(module, state_dict, prefix, local_metadata) -> None:
        # De-register split weights
        idx_strs = [str(idx) for idx in range(module.n_experts)]
        for idx_str in idx_strs:
            delattr(module, idx_str)
        module._parameters = {n: p for n, p in module._parameters.items() if n not in idx_strs}

        # Re-register consolidated weights
        module.register_parameter("_weights", nn.Parameter(module._weight_data))
        delattr(module, "_weight_data")

    @staticmethod
    def load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        idx_strs = [str(idx) for idx in range(module.n_experts)]
        weights_list = []
        for idx in idx_strs:
            if idx not in state_dict:
                raise ValueError(
                    f"Expected key {idx} missing from state_dict: {list(state_dict)=}."
                )
            weights_list.append(state_dict[idx])
            del state_dict[idx]
        weights_stack = torch.stack(weights_list, dim=0)
        state_dict["_weights"] = weights_stack


class TestStateDictHooks:
    d_model = 128
    n_experts = 4

    def test_hooks(self):
        torch.manual_seed(42)
        # Build the models
        exps = Experts(d_model=self.d_model, n_experts=self.n_experts)
        exps_consol = ExpertsConsolidated(d_model=self.d_model, n_experts=self.n_experts)

        exps_state_dict = exps.state_dict()
        exps_consol_state_dict = exps_consol.state_dict()

        # The keys should match
        assert set(exps_state_dict) == set(exps_consol_state_dict)
        # And check tensor shapes:
        for k, p in exps_state_dict.items():
            p_other = exps_consol_state_dict[k]
            assert p.shape == p_other.shape
            # Sanity check that the params are different:
            assert not torch.allclose(p, p_other)

        # The parameters should not:
        assert set(exps._parameters) != set(exps_consol._parameters)

        # The consolidated state_dict data should be views of the full tensors, so that we're not
        # allocating extra memory:
        consol_weight_data_ptr = exps_consol._weights.data.untyped_storage().data_ptr()
        for w in exps_consol_state_dict.values():
            assert w.untyped_storage().data_ptr() == consol_weight_data_ptr

        # Load the consolidated state dict into the non-consolidated model:
        exps.load_state_dict(exps_consol_state_dict)
        # Load the non-consolidated state dict back into the consolidated model:
        exps_consol.load_state_dict(exps.state_dict())

        # Get the consolidated state dict again and check equality (by value; _weights won't be
        # the same due to the new tensor alloc via stacking)
        exps_consol_state_dict_again = exps_consol.state_dict()
        for k, p in exps_consol_state_dict_again.items():
            p_orig = exps_consol_state_dict[k]
            assert p.shape == p_orig.shape
            torch.testing.assert_close(p, exps_consol_state_dict[k])
