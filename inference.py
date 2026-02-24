from models import init_model
from ptq import init_ptq
from pytorch_quantization import quant_modules
import pytorch_quantization
from utils.quant_utils import calibrate_model, set_bits, set_calibrators, set_scale, objective_function
from utils import init_logger
import argparse
import logging
import pickle
import cma

logger = logging.getLogger("my_app")

def get_args():
    parser = argparse.ArgumentParser(description="Audio model evaluation arguments")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mpnet", "fastspeech2", "conformer", "ast", "ecapa"],
        help="Model name to use",
    )
    parser.add_argument(
        "--ptq",
        type=str,
        required=True,
        choices=["adaround", "noisyquant", "ditas", "hyq", "brecq", "repqvit", "smoothquant", "bc"],
        help="Post-training quantization type",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default="max",
        choices=["max", "percentile", "mse", "entropy", "mae", "sqnr", "wmse"],
        help="Calibration type",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Percentile for calibration percentile type",
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=8,
        help="Number of bits for quantization (default: 8)",
    )
    parser.add_argument(
        "--num_calib_data",
        type=int,
        default=100,
        help="Number of calibration samples (default: 100)",
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=100, #8
        help="Number of train samples (default: 10)",
    )
    parser.add_argument(
        "--precompute_scale",
        type=str,
        default=None,
        help="Path to precompute scale (default: None)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="If we export and benchmark the INT8 model",
    )
    #esc
    parser.add_argument(
        "--esc",
        action="store_true",
        help="If we use esc global optimization",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Sigma initial value (default: 0.1)",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=100,
        help="Number of evaluation function evaluated (default: 100)",
    )
    #adaround
    parser.add_argument(
        "--iter_adaround",
        type=int,
        default=50,
        help="Number of iteration for adaround samples (default: 10)",
    )
    #noisyquant
    parser.add_argument(
        "--search_space_mean_noisyquant",
        type=int,
        default=200, #200
        help="Number of iteration for noisyquant mean samples (default: 200)",
    )
    parser.add_argument(
        "--search_space_range_noisyquant",
        type=int,
        default=1000, #1000
        help="Number of iteration for noisyquant range samples (default: 1000)",
    )
    #ditas
    parser.add_argument(
        "--iter_rank_ditas",
        type=int,
        default=10, #200
        help="Number of iteration for ditas low rank decompo (default: 10)",
    )
    parser.add_argument(
        "--reduced_rank_ditas",
        type=int,
        default=32, #1000
        help="Rank of reduced matrix (default: 32)",
    )
    parser.add_argument(
        "--grid_ditas",
        type=int,
        default=21, #1000
        help="Grid size for Ditas optimization (default: 21)",
    )
    #hyq
    parser.add_argument(
        "--iter_hyq",
        type=int,
        default=1000, #200
        help="Number of iteration for hyq optimization (default: 1000)",
    )
    parser.add_argument(
        "--lr_hyq",
        type=float,
        default=1e-2, #1000
        help="Lr for hyq optimization (default: 1e-2)",
    )
    parser.add_argument(
        "--int_softmax",
        action="store_true",
        help="Enable int Softmax"
    )
    #brecq
    parser.add_argument(
        "--iter_brecq",
        type=int,
        default=100, #200
        help="Number of iteration for brecq optimization (default: 1000)",
    )
    parser.add_argument(
        "--lr_brecq",
        type=float,
        default=1e-3, #1000
        help="Lr for brecq optimization (default: 1e-2)",
    )
    parser.add_argument(
        "--p_brecq",
        type=float,
        default=2.4, #1000
        help="p for lp loss for brecq optimization (default: 2.4)",
    )
    #smoothquant
    parser.add_argument(
        "--alpha_smoothquant",
        type=float,
        default=0.5, #1000
        help="alpha for smoothquant optimization (default: 0.5)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    init_logger(args)

    model_name = args.model
    ptq = args.ptq
    num_bits = args.num_bits
    num_calib_data = args.num_calib_data
    num_train_data = args.num_train_data

    #Evaluation FP32
    model_ptq = init_model(model_name)
    model_ptq.load()
    #model_ptq.evaluate()
    #print(model_ptq.model)

    #Evaluation INT8 without calibration
    quant_modules.initialize()
    model_ptq = init_model(model_name)
    model_ptq.load()
    set_bits(model_ptq, num_bits)

    #model_ptq.evaluate()

    model_ptq.pre_calibration(ptq)

    if args.precompute_scale is None:
        set_calibrators(model_ptq)
        calibrate_model(
            model_ptq=model_ptq,
            model_name=model_name,
            num_calib_batch=args.num_calib_data, #100
            method=args.calib,
            percentile=args.percentile) #usually max
    elif ptq == "smoothquant":
        calibrate_model(
            model_ptq=model_ptq,
            model_name=model_name,
            num_calib_batch=100, #100
            method="max",
            percentile=args.percentile)

        with open(args.precompute_scale, "rb") as f:
            dict_amax = pickle.load(f)
        
        list_amax = []
        for name, module in model_ptq.model.named_modules():
            if name in dict_amax.keys():
                if hasattr(module, "input_quantizer"):
                    if module.input_quantizer.amax.shape == dict_amax[name].shape:
                        module.input_quantizer.amax = dict_amax[name].item()
                        list_amax.append(dict_amax[name].item())
    else:
        set_calibrators(model_ptq)
        calibrate_model(
            model_ptq=model_ptq,
            model_name=model_name,
            num_calib_batch=2,
            method="percentile")
        
        with open(args.precompute_scale, "rb") as f:
            dict_amax = pickle.load(f)
        
        list_amax = []
        for name, module in model_ptq.model.named_modules():
            if name in dict_amax.keys():
                if hasattr(module, "input_quantizer"):
                    module.input_quantizer.amax = dict_amax[name].item()
                    list_amax.append(dict_amax[name].item())
    
    # logger.info('Evaluation before PTQ')
    # model_ptq.evaluate()

    # assert False

    if args.esc:
        logger.info('ESC global optimization starts')
        assert args.calib == "mse" or mse in args.precompute_scale, "MSE initialization should be use to fully use the paper described method, you could bypass this warning by commenting"

        list_amax = []
        for name, module in model_ptq.model.named_modules():
            if hasattr(module, "input_quantizer"):
                list_amax.append(module.input_quantizer.amax.item())
        
        es = cma.CMAEvolutionStrategy(
                x0=[1]* len(list_amax), #list_amax
                sigma0=args.sigma,
            )

        es.opts.set({'maxfevals': args.gamma})

        while not es.stop():
            solutions = es.ask()
            train_losses = [objective_function(model_ptq, x, list_amax) for x in solutions]
            es.tell(solutions, train_losses)
            es.disp()

        best = es.result.xfavorite
        scale_list = [x * y for x, y in zip(best, list_amax)]
        set_scale(model, scale_list)
        
        model_ptq.evaluate()

            # dict_amax_esc = {}

            # for name, module in model_ptq.model.named_modules():
            #     if hasattr(module, "input_quantizer"):
            #         dict_amax_esc[name] = module.input_quantizer.amax
            
            # with open(f"scale/dict_amax_{model_name}_{num_bits}_esc_{sigma:.3f}_{j}.pkl", "wb") as f:
            #     pickle.dump(dict_amax_esc, f)
    
    if args.export:
        model_ptq.export()
        model_ptq.benchmark(mode="int")

    if ptq == "adaround":
        layers_input = model_ptq.collect_layer_inputs(ptq, layers_to_quantize=[pytorch_quantization.nn.QuantConv1d, pytorch_quantization.nn.QuantConv2d, pytorch_quantization.nn.QuantLinear], num_batches = num_train_data)
        model = init_ptq(model_ptq.model, ptq, layers_input=layers_input, max_iter=args.iter_adaround) #put max iter = 10k
    elif ptq == "noisyquant":
        layers_input = model_ptq.collect_layer_inputs(ptq, layers_to_quantize=[pytorch_quantization.nn.QuantLinear], num_batches = num_train_data)
        model = init_ptq(model_ptq.model, ptq, layers_input=layers_input, search_space_mean=args.search_space_mean_noisyquant, search_space_range=args.search_space_range_noisyquant)
    elif ptq == "ditas":
        layers_input = model_ptq.collect_layer_inputs(ptq, layers_to_quantize=[pytorch_quantization.nn.QuantLinear], num_batches = num_train_data)
        model = init_ptq(model_ptq.model, ptq, layers_input=layers_input, iter_rank=args.iter_rank_ditas, reduced_rank=args.reduced_rank_ditas, grid=args.grid_ditas)
    elif ptq == "hyq":
        layers_input = model_ptq.collect_layer_inputs(ptq, layers_to_quantize=[pytorch_quantization.nn.QuantConv1d, pytorch_quantization.nn.QuantConv2d], num_batches = num_train_data)
        model = init_ptq(model_ptq.model, ptq, layers_input=layers_input, max_iter=args.iter_hyq, lr=args.lr_hyq, int_softmax=args.int_softmax)
    elif ptq == "brecq":
        if model_name == "mpnet":
            layers_to_quantize = ["ffm", "ccm"]
        elif model_name == "fastspeech2":
            layers_to_quantize = ["pos_ffn", "convolutions"]
        elif model_name == "conformer":
            layers_to_quantize = ["ff1", "conv_block", "attention", "ff2"]
        elif model_name == "ast":
            layers_to_quantize = ["mlp", "attn"]
        elif model_name == "ecapa":
            layers_to_quantize = ["se", "attention"]
        layers_input = model_ptq.collect_layer_inputs(ptq, layers_to_quantize=layers_to_quantize, num_batches = num_train_data)
        model = init_ptq(model_ptq.model, ptq, layers_input=layers_input, max_iter=args.iter_brecq, lr=args.lr_brecq, p=args.p_brecq)
    elif ptq == "repqvit":
        model = init_ptq(model_ptq.model, ptq)
    elif ptq == "smoothquant":
        model = init_ptq(model_ptq.model, ptq, alpha=args.alpha_smoothquant)
    elif ptq == "bc":
        layers_input = model_ptq.collect_layer_inputs(ptq, layers_to_quantize=[pytorch_quantization.nn.QuantLinear], num_batches = num_train_data)
        model = init_ptq(model_ptq.model, ptq, layers_input=layers_input)
    
    logger.info('Evaluation after PTQ')
    #model_ptq.evaluate()