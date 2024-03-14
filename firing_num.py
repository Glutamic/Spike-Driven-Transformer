import argparse
import time
import yaml
import json
import os
import numpy as np
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import torch.nn.utils.prune as prune
from torch import quantization,quantize_per_tensor,floor
import copy
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torchvision import transforms
import pandas as pd
import csv
import struct

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
)
from timm.utils import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
import model, dvs_utils

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="imagenet.yml",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Dataset / Model parameters
parser.add_argument(
    "-data-dir",
    metavar="DIR",
    default="",
    help="path to dataset",
)
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="torch/cifar10",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--model",
    default="spikeformer",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "countception")',
)
parser.add_argument(
    "--pooling_stat",
    default="1111",
    type=str,
    help="pooling layers in SPS moduls",
)
parser.add_argument(
    "--TET",
    default=False,
    type=bool,
    help="",
)
parser.add_argument(
    "--TET-means",
    default=1.0,
    type=float,
    help="",
)
parser.add_argument(
    "--TET-lamb",
    default=0.0,
    type=float,
    help="",
)
parser.add_argument(
    "--spike-mode",
    default="lif",
    type=str,
    help="",
)
parser.add_argument(
    "--layer",
    default=4,
    type=int,
    help="",
)
parser.add_argument(
    "--in-channels",
    default=3,
    type=int,
    help="",
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=1000,
    metavar="N",
    help="number of label classes (Model default if None)",
)
parser.add_argument(
    "--time-steps",
    type=int,
    default=4,
    metavar="N",
    help="",
)
parser.add_argument(
    "--num-heads",
    type=int,
    default=8,
    metavar="N",
    help="",
)
parser.add_argument(
    "--patch-size", type=int, default=None, metavar="N", help="Image patch size"
)
parser.add_argument(
    "--mlp-ratio",
    type=int,
    default=4,
    metavar="N",
    help="expand ration of embedding dimension in MLP block",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "-vb",
    "--val-batch-size",
    type=int,
    default=16,
    metavar="N",
    help="input val batch size for training (default: 32)",
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="weight decay (default: 0.0001)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)

# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="step",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=3,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
parser.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
),
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
parser.add_argument(
    "--jsd",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
parser.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="Enable BCE loss w/ Mixup/CutMix use.",
)
parser.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="Threshold for binarizing softened BCE targets (default: None, disabled)",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
parser.add_argument(
    "--remode", type=str, default="const", help='Random erase mode (default: "const")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.0,
    help="mixup alpha, mixup enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=0.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)
parser.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="Turn off mixup after this epoch, disabled if 0 (default: 0)",
)
parser.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)
parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=0.2,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument(
    "--bn-tf",
    action="store_true",
    default=False,
    help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
)
parser.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
parser.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
parser.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
parser.add_argument(
    "--dist-bn",
    type=str,
    default="",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)
parser.add_argument(
    "--split-bn",
    action="store_true",
    help="Enable separate BN layers per augmentation split.",
)

# Model Exponential Moving Average
parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="decay factor for model weights moving average (default: 0.9998)",
)

# Misc
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 1)",
)
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--dvs-aug",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--dvs-trival-aug",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--save-qkv",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1")',
)
parser.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--large-valid",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)
parser.add_argument(
    "--use-smplified-model",
    action="store_true",
    default=False,
    help="use a simplified model for inference",
)

_logger = logging.getLogger("valid")
stream_handler = logging.StreamHandler()
format_str = "%(asctime)s %(levelname)s: %(message)s"
stream_handler.setFormatter(logging.Formatter(format_str))
_logger.addHandler(stream_handler)
_logger.propagate = False


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:1"
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.initial_seed()  # dataloader multi processing
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random_seed(args.seed, args.rank)

    args.dvs_mode = False
    if args.dataset in ["cifar10-dvs-tet", "cifar10-dvs"]:
        args.dvs_mode = True

    model = create_model(
        args.model,
        T=args.time_steps,
        pretrained=args.pretrained,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        pooling_stat=args.pooling_stat,
        img_size_h=args.img_size,
        img_size_w=args.img_size,
        patch_size=args.patch_size,
        embed_dims=args.dim,
        mlp_ratios=args.mlp_ratio,
        in_channels=args.in_channels,
        qkv_bias=False,
        depths=args.layer,
        sr_ratios=1,
        spike_mode=args.spike_mode,
        dvs_mode=args.dvs_mode,
        TET=args.TET,
        simplified=args.use_smplified_model,
    )
    if args.local_rank == 0:
        _logger.info("Creating model")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _logger.info(f"number of params: {n_parameters}")

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = (
            model.num_classes
        )  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
    )

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != "native":
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    if args.resume:
        resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0,
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    if args.distributed:
        if has_apex and use_amp != "native":
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True, find_unused_parameters=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[args.local_rank], find_unused_parameters=True
            )  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # create the train and eval datasets
    dataset_eval = None, None
    if args.dataset == "cifar10-dvs-tet":
        dataset_eval = dvs_utils.DVSCifar10(
            root=os.path.join(args.data_dir, "test"),
            train=False,
        )
    elif args.dataset == "cifar10-dvs":
        dataset = CIFAR10DVS(
            args.data_dir,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
        )
        _, dataset_eval = dvs_utils.split_to_train_test_set(0.9, dataset, 10)
    elif args.dataset == "gesture":
        dataset_eval = DVS128Gesture(
            args.data_dir,
            train=False,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
        )
    else:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            batch_size=args.batch_size,
            # download=True,
        )

    loader_eval = None
    if args.dataset in dvs_utils.DVS_DATASET:
        loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    elif args.dataset == "imagenet" and args.large_valid:
        dataset_eval.transform = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(288),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
            ]
        )
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval)
        loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.val_batch_size,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=args.val_batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=args.pin_mem,
        )

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = "-".join(
            [
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                "data-" + args.dataset.split("/")[-1],
                f"t-{args.time_steps}",
                f"spike-{args.spike_mode}",
            ]
        )
    output_dir = get_outdir(args.output if args.output else "./output/valid", exp_name)
    if args.rank == 0:
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f"{args.model}.log"), "w"
        )
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(logging.INFO)
        _logger.addHandler(file_handler)

    try:
        for name, module in model.named_modules():
            # print(name, "is conv2d: ", isinstance(module, torch.nn.Conv2d), "is linear: ", isinstance(module, torch.nn.Linear))
            # prune_test_and_eval(model, name, loader_eval, validate_loss_fn, args)  # use this to test prune method and eval
            # if isinstance(module, torch.nn.Conv2d):
            if (".attn." in name or ".mlp." in name) and "conv" in name:
                prune.l1_unstructured(module, name='weight', amount=0.15)
                # prune.ln_structured(module, name='weight', n=2, amount=0.05, dim=1)
                prune.remove(module, 'weight')
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.2)
                # prune.ln_structured(module, name='weight', n=2, amount=0.1, dim=1)
                prune.remove(module, 'weight')
        fuse_module(model)
        
        # statistics(model)
        # show_me_the_money(model, args.use_smplified_model)  # use this to show the max and min of the weights
        for quant_bit in [8, 7, 6 ,5, 4, 3, 2]:
            model_copy = copy.deepcopy(model)
            state_dict = quantize(model, bits=8, qkv_bits=8, qkv_bias_bits=7, 
                                proj_bits=5, proj_bias_bits=5, 
                                fc1_bits=8, fc1_bias_bits=5, 
                                fc2_bits=4, fc2_bias_bits=4, 
                                head_bits=8, head_bias_bits=8)
            model_copy.load_state_dict(state_dict)
            # state_dict = proj_quantize(model_copy, bits=5)
            # model_copy.load_state_dict(state_dict)
            # state_dict = fc1_quantize(model_copy, bits=8)
            # model_copy.load_state_dict(state_dict)
            # state_dict = fc2_quantize(model_copy, bits=4)
            # model_copy.load_state_dict(state_dict)
            # state_dict = head_quantize(model_copy, bits=8)
            # model_copy.load_state_dict(state_dict)
            export_to_binary(state_dict)
            # show_me_the_money(model_copy, args.use_smplified_model)
            exit()
            eval_metrics = validate(
                model_copy,
                loader_eval,
                validate_loss_fn,
                args,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
            )
            if args.local_rank == 0:
                _logger.info("top-1:", eval_metrics["top1"])
                with open('record.txt', 'a') as file:
                    file.write("proj_quantize top-1 acc in quant_bit {} is: {}\n".format(quant_bit, eval_metrics["top1"]))
            break
        exit()
        state_dict = qkv_quantize(model, bits=8, use_smplified_model=args.use_smplified_model)
        model.load_state_dict(state_dict)
        state_dict = param_quantize(model, bits=8, use_smplified_model=args.use_smplified_model)
        model.load_state_dict(state_dict)
        if args.distributed and args.dist_bn in ("broadcast", "reduce"):
            if args.local_rank == 0:
                _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == "reduce")
        eval_metrics = validate(
            model,
            loader_eval,
            validate_loss_fn,
            args,
            output_dir=output_dir,
            amp_autocast=amp_autocast,
        )
        if args.local_rank == 0:
            # non_zero_str = json.dumps(eval_metrics["non_zero"], indent=4)
            # firing_rate_str = json.dumps(eval_metrics["firing_rate"], indent=4)
            _logger.info("top-1:", eval_metrics["top1"])
            # _logger.info("non_zero: ")
            # _logger.info(non_zero_str)
            # _logger.info("firing_rate: ")
            # _logger.info(firing_rate_str)
        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
    except KeyboardInterrupt:
        pass


def prune_test_and_eval(model, name, loader, loss_fn, args, output_dir=None, amp_autocast=suppress):
    if (".attn." in name or ".mlp." in name) and "conv" in name:
        for rate in [0.05 + 0.01 * i for i in range(16)]:
            model_copy = copy.deepcopy(model)
            for name_copy, module in model_copy.named_modules():  # 似乎想通过name得到module只能用迭代器遍历，不能用索引
                if name_copy == name:
                    print("True")
                    copy_module = module
                    break
                else:
                    copy_module = None
            if copy_module is None:
                print(f"module {name} not found")
                exit()
            prune.ln_structured(copy_module, name='weight', n=2, amount=rate, dim=1)
            prune.remove(copy_module, 'weight')
            fuse_module(model_copy)
            state_dict = norm_quantize(model_copy, bits=8)
            model_copy.load_state_dict(state_dict)
            eval_metrics = validate(
                model_copy,
                loader,
                loss_fn,
                args,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
            )
            if args.local_rank == 0:
                _logger.info("top-1:", eval_metrics["top1"])
                with open('record.txt', 'a') as file:
                    file.write("{} top-1 acc in rate {} is: {}\n".format(name, rate, eval_metrics["top1"]))


def statistics(model):
    '''
    统计模型中各层的权重分布，将权重的绝对值去零，减去平均值，并将结果保存到csv文件中
    '''
    state_dict = model.state_dict()
    bins = [i * 2**(-4) for i in range(1, 16)] + [i for i in range(1, 9)]
    bins = [-x for x in bins[::-1]] + [0] + bins
    print(bins)
    data_dict = {}
    for key in state_dict:
        data_list = []
        if "head" in key or ((".attn." in key or ".mlp." in key) and "conv" in key):
            data = state_dict[key].data.cpu().numpy().flatten()
            mask = data != 0
            data_filtered = np.abs(data[mask])
            mean_filter_data = np.mean(data_filtered)
            print(f"去掉了{len(data)-len(data_filtered)}个0，{key} 去零后绝对值的平均值为：{mean_filter_data}")
            std_filter_data = data_filtered - mean_filter_data
            hist, _ = np.histogram(std_filter_data, bins)
            for i in range(len(hist)):
                data_list.append(hist[i])
            data_dict[key] = data_list
    with open('statistics_abs.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_dict.keys())
        writer.writerows(zip(*data_dict.values()))
    return


def show_me_the_money(model, use_smplified_model=False):
    state_dict = model.state_dict()
    bins = [i * 2**(-8) for i in range(1, 256)] + [i for i in range(1, 9)]
    bins = [-x for x in bins[::-1]] + [0] + bins
    data_dict = {}
    for key in state_dict:
        data_list = []
        if (".attn." in key or ".mlp." in key) and "conv" in key:
            # with open('statistics.txt', 'a') as file:
            #     file.write(key + '\n')
            print(f"{key}的最大值为", state_dict[key].data.max().item(), f"{key}的最小值为", state_dict[key].data.min().item())
            data = state_dict[key].data.cpu().numpy().flatten()
            mean = np.mean(data)
            mean_abs = np.mean(np.abs(data))
            print(f"{key} 的平均值为：{mean}, 绝对值的平均值为：{mean_abs}")
            hist, bin_edges = np.histogram(data, bins)
            for i in range(len(hist)):
                # print(f"区间 [{bin_edges[i]:.5f}, {bin_edges[i+1]:.5f}] 的频数为：{hist[i]}")
                data_list.append(hist[i])
                # with open('statistics.txt', 'a') as file:
                #     file.write(f"区间 [{bin_edges[i]:.5f}, {bin_edges[i+1]:.5f}] 的频数为：{hist[i]}\n")
            data_dict[key] = data_list
        elif "head" in key:
            # with open('statistics.txt', 'a') as file:
            #     file.write(key + '\n')
            data = state_dict[key].data.cpu().numpy().flatten()
            mean = np.mean(data)
            mean_abs = np.mean(np.abs(data))
            print(f"{key} 的平均值为：{mean}, 绝对值的平均值为：{mean_abs}")
            hist, bin_edges = np.histogram(data, bins)
            for i in range(len(hist)):
                # print(f"区间 [{bin_edges[i]:.5f}, {bin_edges[i+1]:.5f}] 的频数为：{hist[i]}")
                # with open('statistics.txt', 'a') as file:
                #     file.write(f"区间 [{bin_edges[i]:.5f}, {bin_edges[i+1]:.5f}] 的频数为：{hist[i]}\n")
                data_list.append(hist[i])
            data_dict[key] = data_list

    # write to csv file
            
    # with open('statistics.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(data_dict.keys())
    #     writer.writerows(zip(*data_dict.values()))
    # exit()


def calculate_error(original_weight, quantized_weight):
    # 计算绝对误差
    abs_error = torch.abs(original_weight - quantized_weight)
    
    # 计算相对误差
    relative_error = abs_error / torch.abs(original_weight)
    
    return abs_error, relative_error


def bi_complement(n, bits, dec_bits):  # bits表示二进制位数，dec_bits表示小数点后位数
    # 二进制补码
    int_n = int(n * 2 ** dec_bits)
    max_val = 2 ** (bits - 1) - 1
    min_val = -2 ** (bits - 1)
    if int_n > max_val:
        int_n = max_val
    elif int_n < min_val:
        int_n = min_val
    
    if n >= 0:
        return bin(int_n)[2:].zfill(bits)
    else:
        return bin(2 ** bits + int_n)[2:].zfill(bits)


def export_to_binary(state_dict, folder='binary_files', prefix='quantized'):
    os.makedirs(folder, exist_ok=True)

    for key in state_dict:
        if "head" in key or ((".attn." in key or ".mlp." in key) and "conv" in key):
            filename = os.path.join(folder, f'{prefix}_{key}.txt')
            # size = state_dict[key].data.size()
            data = state_dict[key].cpu().numpy().flatten()
            data = data.tolist()
            with open(filename, 'w') as f:
                for i in range(len(data)):
                    if "q_" in key or "k_" in key or "v_" in key:
                        if "weight" in key:
                            str = bi_complement(data[i], 8, 8)
                        else:
                            str = bi_complement(data[i], 8, 7)
                    elif "fc1_" in key:
                        if "weight" in key:
                            str = bi_complement(data[i], 8, 8)
                        else:
                            str = bi_complement(data[i], 8, 5)
                    elif "proj_" in key:
                        str = bi_complement(data[i], 8, 5)
                    elif "fc2_" in key:
                        str = bi_complement(data[i], 8, 4)
                    elif "head_" in key:
                        str = bi_complement(data[i], 8, 8)
                    f.write(str)
                    f.write('\n')


def  norm_quantize(model, bits=8):
    state_dict = model.state_dict()
    quant_range = 2 ** bits
    for key in state_dict:
        if "head" in key or ((".attn." in key or ".mlp." in key) and "conv" in key):
            state_dict[key].data = floor(quant_range * state_dict[key].data)/quant_range
    return state_dict


def log_quantize(model, key):
    state_dict = model.state_dict()
    # 提取参数的正负号
    sign = torch.sign(state_dict[key])
    # 提取非零元素的索引
    non_zero_indices = state_dict[key].data != 0
    # 计算log2
    state_dict[key].data = torch.where(non_zero_indices, 2 * torch.log2(torch.abs(state_dict[key].data)), state_dict[key].data)
    # 四舍五入
    state_dict[key].data = torch.round(state_dict[key].data)
    # 截断到0到8的范围内
    state_dict[key].data = torch.clamp(state_dict[key].data, -15, 15)
    # 恢复量化
    state_dict[key].data = torch.where(non_zero_indices, 2 ** (state_dict[key].data / 2) * sign, state_dict[key].data)
    return state_dict[key].data


def quantize(model, bits=8, qkv_bits=8, qkv_bias_bits=7, proj_bits=5, proj_bias_bits=5, fc1_bits=8, fc1_bias_bits=5, fc2_bits=4, fc2_bias_bits=4, head_bits=8, head_bias_bits=8, use_smplified_model=False):
    state_dict = model.state_dict()
    for key in state_dict:
        if "head" in key or ((".attn." in key or ".mlp." in key) and "conv" in key):
            if "q_" in key or "k_" in key or "v_" in key:
                if use_smplified_model and "block.0.attn.v_conv" in key:
                        continue
                if "weight" in key:
                    quant_range = 2 ** qkv_bits
                    max = 2 ** (bits - qkv_bits) - 2 ** (-qkv_bits)
                    min = -2 ** (bits - qkv_bits) + 2 ** (-qkv_bits)
                else:
                    quant_range = 2 ** qkv_bias_bits
                    max = 2 ** (bits - qkv_bias_bits) - 2 ** (-qkv_bias_bits)
                    min = -2 ** (bits - qkv_bias_bits) + 2 ** (-qkv_bias_bits)
                state_dict[key].clamp_(min=min, max=max)  # 根据小数位数截断整数范围
                state_dict[key].data = floor(quant_range * state_dict[key].data)/quant_range   #权重量化
            elif "proj_conv" in key:
                if "weight" in key:
                    quant_range = 2 ** proj_bits
                    max = 2 ** (bits - proj_bits) - 2 ** (-proj_bits)
                    min = -2 ** (bits - proj_bits) + 2 ** (-proj_bits)
                else:
                    quant_range = 2 ** proj_bias_bits
                    max = 2 ** (bits - proj_bias_bits) - 2 ** (-proj_bias_bits)
                    min = -2 ** (bits - proj_bias_bits) + 2 ** (-proj_bias_bits)
                state_dict[key].clamp_(min=min, max=max)  # 根据小数位数截断整数范围
                state_dict[key].data = floor(quant_range * state_dict[key].data)/quant_range   #权重量化
            elif "fc1_" in key:
                if "weight" in key:
                    quant_range = 2 ** fc1_bits
                    max = 2 ** (bits - fc1_bits) - 2 ** (-fc1_bits)
                    min = -2 ** (bits - fc1_bits) + 2 ** (-fc1_bits)
                else:
                    quant_range = 2 ** fc1_bias_bits
                    max = 2 ** (bits - fc1_bias_bits) - 2 ** (-fc1_bias_bits)
                    min = -2 ** (bits - fc1_bias_bits) + 2 ** (-fc1_bias_bits)
                state_dict[key].clamp_(min=min, max=max)  # 根据小数位数截断整数范围
                state_dict[key].data = floor(quant_range * state_dict[key].data)/quant_range   #权重量化
            elif "fc2_" in key:
                if "weight" in key:
                    quant_range = 2 ** fc2_bits
                    max = 2 ** (bits - fc2_bits) - 2 ** (-fc2_bits)
                    min = -2 ** (bits - fc2_bits) + 2 ** (-fc2_bits)
                else:
                    quant_range = 2 ** fc2_bias_bits
                    max = 2 ** (bits - fc2_bias_bits) - 2 ** (-fc2_bias_bits)
                    min = -2 ** (bits - fc2_bias_bits) + 2 ** (-fc2_bias_bits)
                state_dict[key].clamp_(min=min, max=max)  # 根据小数位数截断整数范围
                state_dict[key].data = floor(quant_range * state_dict[key].data)/quant_range   #权重量化
            elif "head" in key:
                if "weight" in key:
                    quant_range = 2 ** head_bits
                    max = 2 ** (bits - head_bits) - 2 ** (-head_bits)
                    min = -2 ** (bits - head_bits) + 2 ** (-head_bits)
                else:
                    quant_range = 2 ** head_bias_bits
                    max = 2 ** (bits - head_bias_bits) - 2 ** (-head_bias_bits)
                    min = -2 ** (bits - head_bias_bits) + 2 ** (-head_bias_bits)
                state_dict[key].clamp_(min=min, max=max)  # 根据小数位数截断整数范围
                state_dict[key].data = floor(quant_range * state_dict[key].data)/quant_range   #权重量化
    return state_dict



def proj_quantize(model, bits=8):
    state_dict = model.state_dict()
    quant_range = 2 ** bits
    max = 2 ** (8 - bits) - 2 ** (-bits)
    min = -2 ** (8 - bits) + 2 ** (-bits)
    # state_dict['block.0.attn.talking_heads.weight'].data = floor(quant_range * state_dict['block.0.attn.talking_heads.weight'].data)/quant_range   #权重量化
    state_dict['block.0.attn.proj_conv.weight'].data = floor(quant_range * state_dict['block.0.attn.proj_conv.weight'].data)/quant_range   #权重量化
    state_dict['block.0.attn.proj_conv.bias'].data = floor(quant_range * state_dict['block.0.attn.proj_conv.bias'].data)/quant_range   #权重量化
    return state_dict

def fc1_quantize(model, bits=8):
    state_dict = model.state_dict()
    quant_range = 2 ** bits
    max = 2 ** (8 - bits) - 2 ** (-bits)
    min = -2 ** (8 - bits) + 2 ** (-bits)
    state_dict['block.0.mlp.fc1_conv.weight'].data = floor(quant_range * state_dict['block.0.mlp.fc1_conv.weight'].data)/quant_range   #权重量化
    # state_dict['block.0.mlp.fc1_conv.weight'].data = log_quantize(model, 'block.0.mlp.fc1_conv.weight')   # log量化，只需要4bit+一个符号位
    state_dict['block.0.mlp.fc1_conv.bias'].data = floor(quant_range * state_dict['block.0.mlp.fc1_conv.bias'].data)/quant_range   #权重量化
    return state_dict

def fc2_quantize(model, bits=8):
    state_dict = model.state_dict()
    quant_range = 2 ** bits
    max = 2 ** (8 - bits) - 2 ** (-bits)
    min = -2 ** (8 - bits) + 2 ** (-bits)
    state_dict['block.0.mlp.fc2_conv.weight'].data = floor(quant_range * state_dict['block.0.mlp.fc2_conv.weight'].data)/quant_range   #权重量化
    state_dict['block.0.mlp.fc2_conv.bias'].data = floor(quant_range * state_dict['block.0.mlp.fc2_conv.bias'].data)/quant_range   #权重量化
    return state_dict

def head_quantize(model, bits=8):
    state_dict = model.state_dict()
    quant_range = 2 ** bits
    max = 2 ** (8 - bits) - 2 ** (-bits)
    min = -2 ** (8 - bits) + 2 ** (-bits)
    state_dict['head.weight'].data = floor(quant_range * state_dict['head.weight'].data)/quant_range   #权重量化
    # state_dict['head.weight'].data = log_quantize(model, 'head.weight')   # log量化
    state_dict['head.bias'].data = floor(quant_range * state_dict['head.bias'].data)/quant_range   #权重量化
    # state_dict['head.bias'].data = log_quantize(model, 'head.bias')   # log量化
    return state_dict


def validate(
    model, loader, loss_fn, args, output_dir=None, amp_autocast=suppress, log_suffix=""
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    def calc_non_zero_rate(s_dict, nz_dict, idx, t):
        for k, v_ in s_dict.items():
            v = v_[t, ...]
            x_shape = torch.tensor(list(v.shape))
            all_neural = torch.prod(x_shape)
            z = torch.nonzero(v)
            if k in nz_dict.keys():
                nz_dict[k] += (z.shape[0] / all_neural).item() / idx
            else:
                nz_dict[k] = (z.shape[0] / all_neural).item() / idx
        return nz_dict

    def calc_firing_rate(s_dict, fr_dict, idx, t):
        for k, v_ in s_dict.items():
            v = v_[t, ...]
            if k in fr_dict.keys():
                fr_dict[k] += v.mean().item() / idx
            else:
                fr_dict[k] = v.mean().item() / idx
        return fr_dict

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    fr_dict, nz_dict = {"t0": dict(), "t1": dict(), "t2": dict(), "t3": dict()}, {
        "t0": dict(),
        "t1": dict(),
        "t2": dict(),
        "t3": dict(),
    }
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
            target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output, firing_dict = model(input, hook=dict())
                if args.save_qkv and args.local_rank == 0:
                    torch.save(
                        firing_dict, os.path.join(output_dir, f"qkv_{batch_idx}.pkl")
                    )

            for t in range(args.time_steps):
                fr_single_dict = calc_firing_rate(
                    firing_dict, fr_dict["t" + str(t)], last_idx, t
                )
                fr_dict["t" + str(t)] = fr_single_dict
                nz_single_dict = calc_non_zero_rate(
                    firing_dict, nz_dict["t" + str(t)], last_idx, t
                )
                nz_dict["t" + str(t)] = nz_single_dict

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )

    metrics = OrderedDict(
        [
            ("loss", losses_m.avg),
            ("top1", top1_m.avg),
            ("top5", top5_m.avg),
            ("non_zero", nz_dict),
            ("firing_rate", fr_dict),
        ]
    )

    return metrics


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


def fuse(conv,bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)


if __name__ == "__main__":
    main()
