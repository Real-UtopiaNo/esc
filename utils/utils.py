import torch
import logging
import os
import torch_tensorrt

def init_logger(args):
    logger = logging.getLogger("my_app")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()

    if args.calib == "percentile":
        calib_suffix = f"{args.calib}_{args.percentile}"
    else:
        calib_suffix = args.calib

    # File handler
    if args.ptq == "adaround":
        log_filename = f"logs/{args.ptq}/{args.model}_{args.ptq}_{args.num_bits}_{args.iter_adaround}_{calib_suffix}.log"
    elif args.ptq == "noisyquant":
        log_filename = f"logs/{args.ptq}/{args.model}_{args.ptq}_{args.num_bits}_{args.search_space_mean_noisyquant}_{args.search_space_range_noisyquant}_{calib_suffix}.log"
    elif args.ptq == "ditas":
        log_filename = f"logs/{args.ptq}/{args.model}_{args.ptq}_{args.num_bits}_{args.iter_rank_ditas}_{args.reduced_rank_ditas}_{args.grid_ditas}_{calib_suffix}.log"
    elif args.ptq == "hyq":
        log_filename = f"logs/{args.ptq}/{args.model}_{args.ptq}_{args.num_bits}_{args.iter_hyq}_{args.lr_hyq}_{args.int_softmax}_{calib_suffix}.log"
    elif args.ptq == "brecq":
        log_filename = f"logs/{args.ptq}/{args.model}_{args.ptq}_{args.num_bits}_{args.iter_brecq}_{args.lr_brecq}_{args.p_brecq}_{calib_suffix}.log"
    elif args.ptq == "smoothquant":
        log_filename = f"logs/{args.ptq}/{args.model}_{args.ptq}_{args.num_bits}_{args.alpha_smoothquant}_{calib_suffix}.log"
    else:
        log_filename = f"logs/{args.model}_{args.num_bits}_{calib_suffix}.log"
    file_handler = logging.FileHandler(log_filename, mode='w')

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    #logger.info("This goes to console and file")

def _load_or_compile(module, compile_spec, path):
    if os.path.exists(path):
        print(f"Loading {path}")
        return torch.jit.load(path).eval()
    else:
        print(f"Compiling {path}")
        trt_mod = torch_tensorrt.compile(module, **compile_spec)
        torch.jit.save(trt_mod, path)
        return trt_mod.eval()