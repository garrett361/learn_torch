from dtest import DTest
import torch
import torch.distributed as dist


class TestDeviceMesh(DTest):
    requires_cuda_env = False
    default_world_size=8
    def test_basic(self) -> None:
        tp_dim = 4
        fsdp_dim = self.world_size // tp_dim
        mesh = dist.device_mesh.init_device_mesh(device_type="cpu", mesh_shape=(fsdp_dim, tp_dim), mesh_dim_names=("fsdp", "tp"))
        assert mesh.get_group(0).size() == mesh.get_group("fsdp").size() == fsdp_dim
        assert mesh.get_group(1).size() == mesh.get_group("tp").size() == tp_dim
        assert mesh.size() == tp_dim * fsdp_dim
        assert mesh.ndim==2
        assert mesh.get_rank() == self.rank
        fsdp_rank, tp_rank = divmod(self.rank, tp_dim)
        assert mesh.get_local_rank("tp") == tp_rank
        assert mesh.get_local_rank("fsdp") == fsdp_rank
        assert mesh.get_coordinate() == [fsdp_rank, tp_rank], f"{mesh.get_coordinate()=}, {[fsdp_rank, tp_rank]=}"

        t = torch.randn(16)
        dist.all_reduce(t, group=mesh.get_group(0))
        print(t)
        assert mesh is not None

