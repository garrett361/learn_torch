import torch
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict
from learn_torch.memory.memory import CUDAMemContext
import gc
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama", action="store_true")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--no_act_ckpt", action="store_true")
    parser.add_argument("--no_optim", action="store_true")
    parser.add_argument("--no_foreach", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--use_reentrant", action="store_true")

    args = parser.parse_args()
    print(f"{args=}")

    model_name = "meta-llama/Llama-3.1-8B" if args.llama else "ibm-fms/Bamba-9B"
    config = AutoConfig.from_pretrained(model_name, use_cache=False)
    mem_context_kwargs = {
        "filter_patterns": (
            "allocated_bytes.all.peak",
            "allocated_bytes.all.current",
        )
    }

    with CUDAMemContext(**mem_context_kwargs) as model_init:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda")
    print(f"Num model params: {sum(p.numel() for p in model.parameters())/1e9} B")
    print(
        f"Weight memory: {sum(p.untyped_storage().nbytes() for p in model.parameters())/2**30} GiB"
    )
    print("\nMODEL INIT")
    for k, v in model_init.after.items():
        print(f"{k}: {v}")

    model.train()
    optim = (
        None
        if args.no_optim
        else torch.optim.AdamW(
            model.parameters(), 1e-7, foreach=False if args.no_foreach else True, fused=args.fused
        )
    )
    if not args.no_act_ckpt:
        gradient_checkpointing_kwargs = {"use_reentrant": args.use_reentrant}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    batch_size = 1
    fwd_mem_dict = defaultdict(list)
    bwd_mem_dict = defaultdict(list)
    opt_mem_dict = defaultdict(list)
    try:
        while True:
            batch = torch.randint(10, size=(batch_size, args.seq_len), device="cuda")

            with CUDAMemContext(**mem_context_kwargs) as train_fwd:
                loss = model(input_ids=batch, labels=batch).loss
            print(f"\nFWD: {batch_size=}")
            for k, v in train_fwd.after.items():
                print(f"{k}: {v:.1f}")
                fwd_mem_dict[k].append(f"{v:.1f}")

            with CUDAMemContext(**mem_context_kwargs) as train_bwd:
                loss.backward()
            print(f"\nBWD: {batch_size=}")
            for k, v in train_bwd.after.items():
                print(f"{k}: {v:.1f}")
                bwd_mem_dict[k].append(f"{v:.1f}")

            if args.no_optim:
                model.zero_grad()
            else:
                with CUDAMemContext(**mem_context_kwargs) as step_and_zero:
                    optim.step()
                    optim.zero_grad()
                print(f"\nOPT: {batch_size=}")
                for k, v in step_and_zero.after.items():
                    print(f"{k}: {v:.1f}")
                    opt_mem_dict[k].append(f"{v:.1f}")

            del loss
            torch.cuda.empty_cache()
            gc.collect()

            print(f"\nSucceeded on {batch_size=}.")
            batch_size += 1
    except torch.OutOfMemoryError as e:
        print(f"\nFailed on {batch_size=}")
        for k, v in fwd_mem_dict.items():
            print(f"FWD, {k:>26}: {v}")
        for k, v in bwd_mem_dict.items():
            print(f"BWD, {k:>26}: {v}")
        if not args.no_optim:
            for k, v in opt_mem_dict.items():
                print(f"OPT, {k:>26}: {v}")
        raise e
