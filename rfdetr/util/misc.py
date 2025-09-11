# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict, deque
from typing import Optional, List

import torch
import torch.distributed as dist
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor

if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t", wandb_logging=False, verbose_logging=False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.verbose_logging = verbose_logging
        if wandb_logging:
            import wandb
            self.wandb = wandb
        else:
            self.wandb = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        # Group metrics by category
        main_losses = []
        decoder_0_losses = []
        decoder_1_losses = []
        decoder_2_losses = []
        encoder_losses = []
        unscaled_losses = []
        unscaled_main = []
        unscaled_decoder_0 = []
        unscaled_decoder_1 = []
        unscaled_decoder_2 = []
        unscaled_encoder = []
        other_metrics = []
        
        for name, meter in self.meters.items():
            metric_str = "{}: {}".format(name, str(meter))
            
            if name.endswith('_unscaled'):
                unscaled_losses.append(metric_str)
                # Further categorize unscaled metrics
                base_name = name[:-9]  # Remove '_unscaled'
                if base_name.endswith('_0'):
                    unscaled_decoder_0.append(metric_str)
                elif base_name.endswith('_1'):
                    unscaled_decoder_1.append(metric_str)
                elif base_name.endswith('_2'):
                    unscaled_decoder_2.append(metric_str)
                elif base_name.endswith('_enc'):
                    unscaled_encoder.append(metric_str)
                else:
                    unscaled_main.append(metric_str)
            elif name.endswith('_0'):
                decoder_0_losses.append(metric_str)
            elif name.endswith('_1'):
                decoder_1_losses.append(metric_str)
            elif name.endswith('_2'):
                decoder_2_losses.append(metric_str)
            elif name.endswith('_enc'):
                encoder_losses.append(metric_str)
            elif name in ['loss', 'loss_ce', 'loss_bbox', 'loss_giou', 'class_error']:
                main_losses.append(metric_str)
            else:
                other_metrics.append(metric_str)
        
        # Build categorized output
        result_parts = []
        
        # Main metrics (lr, time, etc.)
        if other_metrics:
            result_parts.append(self.delimiter.join(other_metrics))
        
        # Main losses
        if main_losses:
            result_parts.append("Main: " + self.delimiter.join(main_losses))
        
        # Decoder layers
        if decoder_0_losses:
            result_parts.append("Dec0: " + self.delimiter.join(decoder_0_losses))
        if decoder_1_losses:
            result_parts.append("Dec1: " + self.delimiter.join(decoder_1_losses))
        if decoder_2_losses:
            result_parts.append("Dec2: " + self.delimiter.join(decoder_2_losses))
        
        # Encoder
        if encoder_losses:
            result_parts.append("Enc: " + self.delimiter.join(encoder_losses))
        
        # Unscaled metrics - detailed or summary
        if unscaled_losses:
            if self.verbose_logging:
                # Detailed unscaled output
                unscaled_parts = []
                if unscaled_main:
                    unscaled_parts.append("  ├─ Main: " + self.delimiter.join(unscaled_main))
                if unscaled_decoder_0:
                    unscaled_parts.append("  ├─ Dec0: " + self.delimiter.join(unscaled_decoder_0))
                if unscaled_decoder_1:
                    unscaled_parts.append("  ├─ Dec1: " + self.delimiter.join(unscaled_decoder_1))
                if unscaled_decoder_2:
                    unscaled_parts.append("  ├─ Dec2: " + self.delimiter.join(unscaled_decoder_2))
                if unscaled_encoder:
                    unscaled_parts.append("  └─ Enc: " + self.delimiter.join(unscaled_encoder))
                
                # Fix the last item to use └─
                if len(unscaled_parts) > 0:
                    unscaled_parts[-1] = unscaled_parts[-1].replace("├─", "└─")
                
                unscaled_str = "Unscaled (" + str(len(unscaled_losses)) + " metrics):\n" + "\n".join(unscaled_parts)
                result_parts.append(unscaled_str)
            else:
                # Summary unscaled output
                result_parts.append("Unscaled: " + str(len(unscaled_losses)) + " metrics (use verbose_logging=True to expand)")
        
        if not result_parts:
            return ""
        
        # Format the tree structure
        if len(result_parts) == 1:
            return "\n└─ " + result_parts[0]
        else:
            return "\n├─ " + "\n├─ ".join(result_parts[:-1]) + "\n└─ " + result_parts[-1]

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # Header format for the first line
        if torch.cuda.is_available():
            header_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            header_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.wandb:
                    if is_main_process():
                        log_dict = {k: v.value for k, v in self.meters.items()}
                        self.wandb.log(log_dict)
                # Print header line
                if torch.cuda.is_available():
                    header_line = header_msg.format(
                        i, len(iterable), eta=eta_string,
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                else:
                    header_line = header_msg.format(
                        i, len(iterable), eta=eta_string,
                        time=str(iter_time), data=str(data_time))
                
                # Print header + metrics in multi-line format
                metrics_str = str(self)
                print(header_line + metrics_str)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(obj, f, *args, **kwargs):
    """
    Safely save objects, removing any callbacks that can't be pickled
    """
    if is_main_process():
        torch.save(obj, f, *args, **kwargs)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__.split(".")[1]) < 7.0:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def strip_checkpoint(checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    new_state_dict = {
        'model': state_dict['model'],
        'args': state_dict['args'],
    }
    torch.save(new_state_dict, checkpoint)


def format_test_results(results_dict, verbose=False):
    """
    Format test results into a structured, readable output.
    
    Args:
        results_dict (dict): Test results dictionary
        verbose (bool): Whether to show detailed unscaled metrics
    
    Returns:
        str: Formatted test results string
    """
    if not results_dict:
        return "No test results available"
    
    # Extract different types of metrics
    loss_metrics = {}
    unscaled_metrics = {}
    eval_results = results_dict.get('results_json', {})
    coco_eval = results_dict.get('coco_eval_bbox', [])
    
    # Categorize metrics
    for key, value in results_dict.items():
        if key in ['results_json', 'coco_eval_bbox']:
            continue
        elif key.endswith('_unscaled'):
            unscaled_metrics[key] = value
        elif 'loss' in key or 'class_error' in key:
            loss_metrics[key] = value
    
    # Group loss metrics by category
    main_losses = {}
    decoder_0_losses = {}
    decoder_1_losses = {}
    decoder_2_losses = {}
    encoder_losses = {}
    
    for key, value in loss_metrics.items():
        if key.endswith('_0'):
            decoder_0_losses[key] = value
        elif key.endswith('_1'):
            decoder_1_losses[key] = value
        elif key.endswith('_2'):
            decoder_2_losses[key] = value
        elif key.endswith('_enc'):
            encoder_losses[key] = value
        else:
            main_losses[key] = value
    
    # Build formatted output
    lines = ["", "Test Results Summary:", "=" * 50]
    
    # Performance Metrics
    if eval_results:
        lines.append("Performance Metrics:")
        map_val = eval_results.get('map', -1.0)
        precision = eval_results.get('precision', float('nan'))
        recall = eval_results.get('recall', 0.0)
        
        if map_val >= 0:
            lines.append(f"├─ mAP@[0.5:0.95]: {map_val:.3f}")
        else:
            lines.append("├─ mAP@[0.5:0.95]: N/A (no predictions)")
            
        if not (precision != precision):  # Check if not NaN
            lines.append(f"├─ Precision: {precision:.3f}")
        else:
            lines.append("├─ Precision: N/A (no predictions)")
            
        lines.append(f"└─ Recall: {recall:.3f}")
        
        # Class-specific results
        class_map = eval_results.get('class_map', [])
        if class_map and len(class_map) > 0:
            lines.append("")
            lines.append("Per-Class Results:")
            for i, cls_result in enumerate(class_map):
                cls_name = cls_result.get('class', f'class_{i}')
                cls_map = cls_result.get('map@50:95', -1.0)
                cls_map50 = cls_result.get('map@50', -1.0)
                if i == len(class_map) - 1:
                    prefix = "└─"
                else:
                    prefix = "├─"
                    
                if cls_map >= 0:
                    lines.append(f"{prefix} {cls_name}: mAP@[0.5:0.95]={cls_map:.3f}, mAP@0.5={cls_map50:.3f}")
                else:
                    lines.append(f"{prefix} {cls_name}: No predictions")
    
    lines.append("")
    
    # Loss Metrics
    lines.append("Loss Metrics:")
    
    def format_losses(losses_dict, name):
        if not losses_dict:
            return []
        loss_strs = []
        for key, value in losses_dict.items():
            if isinstance(value, float):
                loss_strs.append(f"{key}: {value:.4f}")
            else:
                loss_strs.append(f"{key}: {value}")
        return [f"├─ {name}: {' │ '.join(loss_strs)}"]
    
    # Add loss categories
    lines.extend(format_losses(main_losses, "Main"))
    lines.extend(format_losses(decoder_0_losses, "Dec0"))
    lines.extend(format_losses(decoder_1_losses, "Dec1"))
    lines.extend(format_losses(decoder_2_losses, "Dec2"))
    lines.extend(format_losses(encoder_losses, "Enc"))
    
    # Unscaled metrics
    if unscaled_metrics:
        if verbose:
            lines.append("├─ Unscaled Metrics:")
            # Group unscaled metrics similarly
            unscaled_main = {}
            unscaled_dec0 = {}
            unscaled_dec1 = {}
            unscaled_dec2 = {}
            unscaled_enc = {}
            
            for key, value in unscaled_metrics.items():
                base_name = key[:-9]  # Remove '_unscaled'
                if base_name.endswith('_0'):
                    unscaled_dec0[key] = value
                elif base_name.endswith('_1'):
                    unscaled_dec1[key] = value
                elif base_name.endswith('_2'):
                    unscaled_dec2[key] = value
                elif base_name.endswith('_enc'):
                    unscaled_enc[key] = value
                else:
                    unscaled_main[key] = value
            
            for losses_dict, name in [(unscaled_main, "Main"), (unscaled_dec0, "Dec0"), 
                                    (unscaled_dec1, "Dec1"), (unscaled_dec2, "Dec2"), 
                                    (unscaled_enc, "Enc")]:
                if losses_dict:
                    loss_strs = []
                    for key, value in losses_dict.items():
                        if isinstance(value, float):
                            loss_strs.append(f"{key}: {value:.4f}")
                        else:
                            loss_strs.append(f"{key}: {value}")
                    lines.append(f"│  ├─ {name}: {' │ '.join(loss_strs)}")
            
            # Fix the last unscaled item
            if lines[-1].startswith("│  ├─"):
                lines[-1] = lines[-1].replace("│  ├─", "│  └─")
        else:
            lines.append(f"└─ Unscaled: {len(unscaled_metrics)} metrics (use verbose=True to expand)")
    
    # Fix the last main item if unscaled is not shown or is verbose
    if lines and lines[-1].startswith("├─") and not verbose:
        lines[-1] = lines[-1].replace("├─", "└─")
    
    lines.append("")
    return "\n".join(lines)