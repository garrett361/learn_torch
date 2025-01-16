import torch
from transformers import AutoModelForCausalLM, AutoConfig
from memory import CUDAMemContext

if __name__ == "__main__":
    model_name = "ibm-fms/Bamba-9B"
    config = AutoConfig.from_pretrained(model_name)

    with CUDAMemContext() as model_init:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda")
    prefix = ".all."
    for k, v in model_init.after.items():
        if prefix in k:
            print(f"{k.replace(prefix, '')}: {v}")
    batch_size = 1
    seq_len = 4096
    try:
        while True:
            batch = torch.randint(10, size=(batch_size, seq_len), device="cuda")
            with CUDAMemContext() as fwd:
                with torch.no_grad():
                    model(input_ids=batch)
            batch_size += 1
            print(f"\nSucceeded on {batch_size=}. Memory stats after:")
            for k, v in fwd.after.items():
                if prefix in k:
                    print(f"{k.replace(prefix, '.')}: {v}")
    except Exception as e:
        print(f"\nFailed on {batch_size=}")
        raise e
