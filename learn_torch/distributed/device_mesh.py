import pytest
import torch
import torch.distributed as dist

from dtest import DTest


class TestDeviceMesh(DTest):
    requires_cuda_env = False
    default_world_size = 8

    def test_basic(self) -> None:
        tp_dim = 4
        fsdp_dim = self.world_size // tp_dim
        mesh = dist.device_mesh.init_device_mesh(
            device_type="cpu", mesh_shape=(fsdp_dim, tp_dim), mesh_dim_names=("fsdp", "tp")
        )
        assert mesh.get_group(0).size() == mesh.get_group("fsdp").size() == fsdp_dim
        assert mesh.get_group(1).size() == mesh.get_group("tp").size() == tp_dim
        assert mesh.size() == tp_dim * fsdp_dim
        assert mesh.ndim == 2
        assert mesh.get_rank() == self.rank
        fsdp_rank, tp_rank = divmod(self.rank, tp_dim)
        assert mesh.get_local_rank("tp") == tp_rank
        assert mesh.get_local_rank("fsdp") == fsdp_rank
        assert mesh.get_coordinate() == [
            fsdp_rank,
            tp_rank,
        ], f"{mesh.get_coordinate()=}, {[fsdp_rank, tp_rank]=}"

        t = torch.randn(16)
        dist.all_reduce(t, group=mesh.get_group(0))
        print(t)
        assert mesh is not None

    @pytest.mark.world_size(2 * 3 * 4)
    def test_3d_mesh_slicing(self) -> None:
        mesh = dist.device_mesh.init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 3, 4),
            mesh_dim_names=("0", "1", "2"),
        )
        slice_01 = mesh["0", "1"]
        assert slice_01.shape == (2, 3), f"{slice_01.shape=}"
        slice_12 = mesh["1", "2"]
        assert slice_12.shape == (3, 4), f"{slice_12.shape=}"

        slice_01_flat = slice_01._flatten()
        assert slice_01_flat.ndim == 1
        assert slice_01_flat.mesh_dim_names[0] == "0_1"
