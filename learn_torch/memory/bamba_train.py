import torch
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict
from learn_torch.memory.memory import CUDAMemContext
import gc
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ibm-fms/Bamba-9B")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--no_act_ckpt", action="store_true")

    args = parser.parse_args()
    print(f"{args=}")

    config = AutoConfig.from_pretrained(args.model_name)
    mem_context_kwargs = {
        "filter_patterns": (
            "allocated_bytes.all.peak",
            "allocated_bytes.all.current",
        )
    }
    with CUDAMemContext(**mem_context_kwargs) as model_init:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda")
    print("\nMODEL INIT")
    for k, v in model_init.after.items():
        print(f"{k}: {v}")

    model.train()
    if not args.no_act_ckpt:
        model.gradient_checkpointing_enable()

    batch_size = 1
    fwd_mem_dict = defaultdict(list)
    bwd_mem_dict = defaultdict(list)
    try:
        while True:
            batch = torch.randint(10, size=(batch_size, args.seq_len), device="cuda")
            with CUDAMemContext(**mem_context_kwargs) as train_fwd:
                loss = model(input_ids=batch, labels=batch).loss
            with CUDAMemContext(**mem_context_kwargs) as train_bwd:
                loss.backward()
            torch.cuda.synchronize()
            print(f"\nSucceeded on {batch_size=}.")

            print(f"\nFWD: {batch_size=}")
            for k, v in train_bwd.after.items():
                print(f"{k}: {v}")
                fwd_mem_dict[k].append(v)

            print(f"\nBWD: {batch_size=}")
            for k, v in train_bwd.after.items():
                print(f"{k}: {v}")
                bwd_mem_dict[k].append(v)

            del loss
            torch.cuda.empty_cache()
            gc.collect()
            model.zero_grad()
            batch_size += 1
    except Exception as e:
        print(f"\nFailed on {batch_size=}")
        for k, v in fwd_mem_dict.items():
            print(f"FWD, {k}: {v:.1f}")
        for k, v in bwd_mem_dict.items():
            print(f"BWD, {k}: {v::.1f}")
        raise e
