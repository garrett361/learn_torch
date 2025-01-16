from torch.profiler import profile, record_function, ProfilerActivity


def get_args(*args, **kwargs):
    pass


def get_impl(*args, **kwargs):
    pass


if __name__ == "__main__":
    args = get_args()
    impl = get_impl()
    for warmup in range(5):
        impl(*args)
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            impl(*args)
    print(
        f"{impl.__name__}: ",
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=10),
    )
