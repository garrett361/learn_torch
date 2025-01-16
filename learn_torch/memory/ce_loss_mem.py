import torch
import torch.nn.functional as F
from learn_torch.memory.memory import CUDAMemContext

if __name__ == "__main__":
    vocab_size = 128256  # Bamba
    seq_len = 4096
    batch_size = 1
    numel = vocab_size * seq_len
    with CUDAMemContext() as input_mem:
        logits = torch.randn(
            batch_size, numel, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        labels = torch.randint(vocab_size, size=(batch_size,), device="cuda")
    with CUDAMemContext() as loss_mem:
        loss = F.cross_entropy(logits, labels)
    with CUDAMemContext() as back_mem:
        loss.backward()

    prefix = ".all."
    for n, mc in zip(("inputs", "loss", "backwards"), (input_mem, loss_mem, back_mem)):
        print(f"\n##########   {n}   ##########\n")
        print("AFTER")
        for k, v in mc.after.items():
            if prefix in k and "bytes" in k:
                print(f"{k.replace(prefix, '.').replace('bytes', 'gib')}: {v}")
        print("\nDELTA")
        for k, v in mc.delta.items():
            if prefix in k:
                print(f"{k.replace(prefix, '.').replace('bytes', 'gib')}: {v}")
