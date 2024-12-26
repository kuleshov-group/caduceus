
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl

from transformers import AutoConfig, AutoModelForMaskedLM
from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import Caduceus
from caduceus.modeling_caduceus import CaduceusForSequenceClassification
import json
import os
import random
import time
from functools import wraps
from typing import Callable, List, Sequence
from datetime import datetime
from pathlib import Path

import fsspec
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks
from caduceus.tokenization_caduceus import CaduceusTokenizer
log = src.utils.train.get_logger(__name__)
import psutil
import subprocess

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends
from src.tasks.decoders import SequenceDecoder
seed=42
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))

import psutil
import torch
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional
import time
import threading
import queue
from contextlib import contextmanager

@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage"""
    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None

def load_model(checkpoint_path, config_path):
    # Load config
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['hyper_parameters']
    model = SequenceLightningModule(config)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model

class GPUMonitor:
    """GPU monitoring with fallback methods"""
    
    @staticmethod
    def get_gpu_info(device_id: int = 0) -> Dict[str, float]:
        """Get GPU information using available methods"""
        try:
            # Try using nvidia-smi through subprocess
            result = subprocess.check_output(
                [
                    'nvidia-smi',
                    f'--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,nounits,noheader'
                ],
                encoding='utf-8'
            )
            used_mem, total_mem, util = map(float, result.strip().split(','))
            return {
                'memory_used_gb': used_mem / 1024,  # Convert MB to GB
                'memory_total_gb': total_mem / 1024,
                'utilization': util
            }
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                # Fallback to torch.cuda
                if torch.cuda.is_available():
                    used_mem = torch.cuda.memory_allocated(device_id) / (1024**3)  # Convert bytes to GB
                    total_mem = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
                    return {
                        'memory_used_gb': used_mem,
                        'memory_total_gb': total_mem,
                        'utilization': None  # torch.cuda doesn't provide utilization info
                    }
            except:
                pass
            
            # Return None if no GPU info available
            return {
                'memory_used_gb': None,
                'memory_total_gb': None,
                'utilization': None
            }

class ResourceMonitor:
    """Monitors system resources in a background thread."""
    
    def __init__(self, device_id: int = 0, sampling_interval: float = 0.1):
        self.device_id = device_id
        self.sampling_interval = sampling_interval
        self.snapshots = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self.gpu_monitor = GPUMonitor()
    
    def start(self):
        """Start monitoring in background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        """Stop monitoring and return all snapshots."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        
        # Collect all snapshots
        snapshots = []
        while not self.snapshots.empty():
            snapshots.append(self.snapshots.get())
        return snapshots
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Get CPU stats
                cpu_percent = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory()
                ram_used = ram.used / (1024**3)  # Convert to GB
                ram_total = ram.total / (1024**3)
                
                # Get GPU stats
                gpu_info = self.gpu_monitor.get_gpu_info(self.device_id)
                
                snapshot = ResourceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    ram_used_gb=ram_used,
                    ram_total_gb=ram_total,
                    gpu_memory_used_gb=gpu_info['memory_used_gb'],
                    gpu_memory_total_gb=gpu_info['memory_total_gb'],
                    gpu_utilization=gpu_info['utilization']
                )
                self.snapshots.put(snapshot)
                
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                break

@contextmanager
def timer(name: str, timings: Dict[str, float]):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if name in timings:
            if isinstance(timings[name], list):
                timings[name].append(elapsed)
            else:
                timings[name] = [timings[name], elapsed]
        else:
            timings[name] = elapsed

def get_memory_stats():
    """Get current memory statistics."""
    stats = {
        'cpu': {
            'used_gb': psutil.Process().memory_info().rss / (1024**3),
            'percent': psutil.Process().memory_percent()
        }
    }
    
    if torch.cuda.is_available():
        stats['gpu'] = {
            'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024**3)
        }
    
    return stats

def log_memory_stats(logger, prefix=""):
    """Log current memory statistics."""
    stats = get_memory_stats()
    logger.info(f"{prefix}CPU Memory: {stats['cpu']['used_gb']:.2f} GB ({stats['cpu']['percent']:.1f}%)")
    if 'gpu' in stats:
        logger.info(f"{prefix}GPU Memory: {stats['gpu']['allocated_gb']:.2f} GB allocated, "
                   f"{stats['gpu']['reserved_gb']:.2f} GB reserved")

# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        log.error("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        log.warning(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level: access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        # To be set in `setup`
        self.encoder, self.decoder, self.model = None, None, None
        self.task, self.loss, self.loss_val = None, None, None
        self.metrics, self.train_torchmetrics, self.val_torchmetrics, self.test_torchmetrics = None, None, None, None
        self.setup()

        self._state = None
        self.dataset = "phage_fragment_inphared"
        self.val_loader_names, self.test_loader_names = None, None

    def setup(self, stage=None):
        if not self.hparams.train.disable_dataset:
            print("setting up dataset\n")
            self.dataset.setup()

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more
        # memory than the others.
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        config_path = self.hparams.model.pop("config_path", None)
        if config_path is not None:
            with open(config_path) as f:
                model_config_from_file = json.load(f)
            self.hparams.model.update(model_config_from_file)
            # Check if dropout_layer_norm is compiled
            try:
                from flash_attn.ops.layer_norm import dropout_add_layer_norm
            except ImportError:
                if self.hparams.model.get("fused_dropout_add_ln", None) is not None:
                    self.hparams.model.update({"fused_dropout_add_ln": False})
        # TODO: Hacky way to get complement_map for Caduceus models; need to find a more elegant implementation
        if "caduceus" in self.hparams.model.get("_name_"):
            OmegaConf.update(
                self.hparams.model.config, "complement_map", self.dataset.tokenizer.complement_map, force_add=True
            )
        # Instantiate the config class if using hydra's _target_ paradigm for the config
        if self.hparams.model.get("config", None) is not None and self.hparams.model.config.get("_target_", None) is not None:
            model_hparams = OmegaConf.to_container(self.hparams.model, resolve=True)
            model_hparams["config"] = hydra.utils.instantiate(model_hparams["config"])
            self.model = utils.instantiate(registry.model, model_hparams)
        else:
            self.model = utils.instantiate(registry.model, self.hparams.model)
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # if self.hparams.train.get("compile_model", False):
        #     self.model = torch.compile(self.model, dynamic=False)

        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Extract the modules, so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics

    def load_state_dict(self, state_dict, strict=False):
        if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            state_dict = model_state_hook(self.model, state_dict)

        log.info("Custom load_state_dict function is running.")
    
        model_state = self.state_dict()
        missing_keys = set(model_state.keys()) - set(state_dict.keys())
    
        for key in missing_keys:
            if 'decoder' in key:
                print(f"Loading decoder key from scratch: {key}")
                state_dict[key] = model_state[key]

        return super().load_state_dict(state_dict, strict=strict)


    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
                (n := self.hparams.train.state.n_context) is None
                or isinstance(n, int)
                and n >= 0
        )
        assert (
                (n := self.hparams.train.state.n_context_eval) is None
                or isinstance(n, int)
                and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, training=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if training else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    def forward(self, batch):
        return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t)  # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _shared_step(self, batch, batch_idx, prefix="train"):
        """Shared step logic between training, validation, and test"""
        self._process_state(batch, batch_idx, training=(prefix == "train"))
        x, y, w = self.forward(batch)

        # Loss
        if prefix == 'train':
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)

        # Metrics
        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics
        torchmetrics = getattr(self, f'{prefix}_torchmetrics')
        torchmetrics(x, y, loss=loss)

        log_on_step = 'eval' in self.hparams and self.hparams.eval.get('log_on_step', False) and prefix == 'train'

        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # log the whole dict, otherwise lightning takes the mean to reduce it
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        self.log_dict(
            torchmetrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

    def training_epoch_end(self, outputs):
        # Log training torchmetrics
        super().training_epoch_end(outputs)

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

    def validation_epoch_end(self, outputs):
        # Log all validation torchmetrics
        super().validation_epoch_end(outputs)

    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    def test_epoch_end(self, outputs):
        # Log all test torchmetrics
        super().test_epoch_end(outputs)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so that it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": float(self.current_epoch)}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial
        # sanity check
        ema = (
                self.val_loader_names[dataloader_idx].endswith("/ema")
                and self.optimizers().optimizer.stepped
        )
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):
        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups:", hps)  # TODO: log.info throws error because hps is list of dicts
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        # Layer Decay
        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers:
                    num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (
                        self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizers param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)
        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (e.g., if test is duplicate)
        eval_loader_names = []
        eval_loaders = []
        if not self.hparams.train.get("remove_val_loader_in_eval", False):
            eval_loader_names += val_loader_names
            eval_loaders += val_loaders
        if not self.hparams.train.get("remove_test_loader_in_eval", False):
            eval_loader_names += test_loader_names
            eval_loaders += test_loaders
        return eval_loader_names, eval_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders

class BatchPredictor:
    def __init__(
        self, 
        config_path: str,
        checkpoint_path: str,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Track timings
        self.batch_times = []
        self.memory_usage = []
        
        # Load model
        start_time = time.time()
        self.model = load_model(checkpoint_path, config_path)
                
        self.model.eval()
        self.model.to(device)
        
        self.model_load_time = time.time() - start_time
        
        model_hash = hash(str([(k, v.sum().item()) for k, v in self.model.state_dict().items()]))
        logger.info(f"Model weight hash: {model_hash}")

        # Initialize tokenizer
        logger.info("Initializing CaduceusTokenizer")
        self.tokenizer = CaduceusTokenizer(
            model_max_length=4096,
            add_special_tokens=False
        )
    
    def get_gpu_memory(self):
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024**2
        return 0
    
    def process_batch(self, sequences: List[str]) -> torch.Tensor:
        """Process a batch of sequences."""
        batch_start = time.time()
        #print(f"Processing batch with {len(sequences)} sequences")
        #print("Made it to process batch\n")
        #if not self.model.training:
        #    logger.info("Model is correctly in eval mode")
        #else:
        #    logger.warning("Model was in training mode! Setting to eval mode.")
        #    self.model.eval()

        # Tokenize
        encoded = [
            self.tokenizer(
                seq,
                padding="max_length",
                max_length=self.model.hparams.dataset.max_length,
                truncation=True,
                add_special_tokens=False
            ) for seq in sequences
        ]
        
        # Convert to tensor
        input_ids = torch.stack([
            torch.tensor(enc["input_ids"]) for enc in encoded
        ]).to(self.device)
        
        input_ids = torch.where(
            input_ids == self.tokenizer._vocab_str_to_int["N"],
            self.tokenizer.pad_token_id,
            input_ids
        )
        
        dummy_labels = torch.zeros(len(sequences), 2, device=self.device)
        
        # Model inference
        with torch.no_grad():
            batch = (input_ids, None)  # Model expects (input, target) tuple
            outputs = self.model.forward(batch)
            probs = torch.softmax(outputs[0], dim=-1)

        # Keep your existing metrics logging code
        batch_time = time.time() - batch_start
        self.batch_times.append(batch_time)
        self.memory_usage.append(self.get_gpu_memory())

        return probs
    
    def predict_from_csv(self, input_file: str, output_file: str):
        """Process sequences from CSV and save results with metrics."""
        start_time = time.time()
        print("Made it to predict from csv\n")
        
        # Read input CSV
        logger.info(f"Reading sequences from {input_file}")
        df = pd.read_csv(input_file)
        data_load_time = time.time() - start_time
        
        if 'sequence' not in df.columns:
            raise ValueError("Input CSV must have a 'sequence' column")
        
        all_predictions = []
        peak_memory = 0
        
        # Process in batches
        n_batches = (len(df) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(range(0, len(df), self.batch_size), desc="Processing batches", total=n_batches)

        all_predictions = []
        for i in pbar:
            batch_sequences = df['sequence'].iloc[i:i + self.batch_size].tolist()
            batch_probs = self.process_batch(batch_sequences)
    
            # Get probabilities for both classes
            # Converting to numpy and getting probabilities for both classes (non-phage and phage)
            class_probs = batch_probs.cpu().numpy()
    
            # Convert to list format matching DNABERT2
            batch_predictions = [prob.tolist() for prob in class_probs]
            all_predictions.extend(batch_predictions)
    
            # Update progress bar
            current_memory = self.get_gpu_memory()
            peak_memory = max(peak_memory, current_memory)
            pbar.set_postfix({
                'batch_time': f'{np.mean(self.batch_times[-10:]):.3f}s',
                'gpu_mem': f'{int(current_memory)}MB'
            })

        # Add predictions to dataframe
        df['predictions'] = all_predictions
        df['phage_prob'] = df['predictions'].apply(lambda x: x[1])

        if 'gc' not in df.columns:
            # Add GC content calculation if needed
            df['gc'] = df['sequence'].apply(lambda x: (x.count('G') + x.count('C')) / len(x) * 100)

        # Add start column if needed (assuming sequential numbering)
        if 'start' not in df.columns:
            df['start'] = range(0, len(df) * 500, 500)
        
        # Prepare metrics
        total_time = time.time() - start_time
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'input_file': input_file,
            'num_sequences': len(df),
            'batch_size': self.batch_size,
            'timing': {
                'total_time': total_time,
                'model_load_time': self.model_load_time,
                'data_load_time': data_load_time,
                'batch_time_mean': np.mean(self.batch_times),
                'batch_time_std': np.std(self.batch_times),
                'sequences_per_second': len(df) / total_time
            },
            'memory': {
                'peak_gpu_mb': peak_memory,
                'mean_gpu_mb': np.mean(self.memory_usage),
                'peak_process_gb': psutil.Process().memory_info().rss / (1024**3)
            },
            'predictions': {
                'num_phages': int(sum(df['phage_prob'] > 0.5)),
                'num_non_phages': int(sum(df['phage_prob'] <= 0.5))
            }
        }
        
        # Reorder columns to match format
        df = df[['Seq_Id', 'start', 'gc', 'predictions']]
        
        # Save results
        save_start = time.time()
        df.to_csv(output_file, index=False)
        save_time = time.time() - save_start

        # Save metrics
        metrics_file = str(Path(output_file).with_suffix('.metrics.json'))
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total sequences: {len(df)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average batch time: {metrics['timing']['batch_time_mean']:.3f}s")
        logger.info(f"Sequences per second: {metrics['timing']['sequences_per_second']:.2f}")
        logger.info(f"Peak GPU memory: {metrics['memory']['peak_gpu_mb']:.0f}MB")
        logger.info(f"Predicted positive class: {metrics['predictions']['num_phages']}")
        logger.info(f"Predicted negative class: {metrics['predictions']['num_non_phages']}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch predict phages from CSV')
    parser.add_argument('--config', required=True, help='Path to model config')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device to run on')
    
    args = parser.parse_args()
    
    predictor = BatchPredictor(
        args.config,
        args.checkpoint,
        batch_size=args.batch_size,
        device=args.device
    )
    
    predictor.predict_from_csv(args.input, args.output)

if __name__ == "__main__":
    main()
