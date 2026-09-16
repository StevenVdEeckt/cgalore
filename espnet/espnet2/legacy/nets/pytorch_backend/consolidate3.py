# Copyright 2022 Steven Vander Eeckt

""" Continual Learning extensions for End-to-End ASR (ESPnet2) """

"""  
    consolidate2_from_espnet2.py is meant to compute e.g. importance weights (or whatever else is required)
    after training a task, such that these importance weights (or anything else) can later be used
    to prevent catastrophic forgetting of the given task. 
"""

import numpy as np
import copy
import torch
import torch.nn.functional as F

import logging
from abc import ABC, abstractmethod
import os
import typing

from espnet2.layers.create_adapter_utils import (
    check_target_module_exists,
    get_submodules,
    replace_module,
    get_target_key,
)




"""  
########################### HELP FUNCTIONS ###########################################################################
"""


def _recursive_to(
        xs,
        device
):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple) or isinstance(xs, list):
        return tuple(_recursive_to(x, device) for x in xs)
    if isinstance(xs, dict):
        return {n: _recursive_to(p, device) for n, p in xs.items()}
    return xs


def cuda_overview(
        message: str = ""
):
    """
        Displays CUDA memory information

        :param str message: additional message to display (e.g. location in code)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory * 10.0 ** (-9)
        r = torch.cuda.memory_reserved(0) * 10.0 ** (-9)
        a = torch.cuda.memory_allocated(0) * 10.0 ** (-9)
        f = r - a  # free inside reserved
        logging.info("%s: CUDA memory - "
                     "total: %.4f; reserved: %.4f; allocated: %.4f; available: %.4f" % (message, t, r, a, f))
    else:
        pass


def copy_model(
        model: torch.nn.Module,
        device: str,
        train: bool = False,
        requires_grad: bool = False
):
    """
        Copy the model and make it untrainable

        :param torch.nn.Module model: the model
        :param str device:  cpu or gpu device to map model to
        :param bool train: if False, model is set to eval mode
    """
    # for batch norm and dropout layers
    old_model = copy.deepcopy(model).train(train)
    for name, param in old_model.named_parameters():
        param.requires_grad = requires_grad
    return old_model.to(device)


def low_memory_vmap(func):
    """
    Returns a function that maps `func` over each element in the batch (first dimension),
    similar to torch.vmap(func), but sequentially to reduce CUDA memory pressure.

    Example:
        f = low_memory_vmap(my_func)
        result = f(x)  # where x has shape (batch, ...)
    """
    def wrapper(x):
        results = []
        for i in range(x.size(0)):
            results.append(func(x[i]))
        return torch.stack(results, dim=0)
    return wrapper



"""  
########################### ABSTRACT CLASSES #########################################################################
"""


class Consolidation(ABC):
    """
        Consolidate Abstract Class

        Any such class must have the following methods:
                --> consolidate(speech, speech_lenghts, text, text_lengths, task=None)
                --> save()
    """
    @abstractmethod
    def consolidate(
            self,
            speech: torch.tensor,
            speech_lengths: torch.tensor,
            text: torch.tensor,
            text_lengths: torch.tensor,
            text_prev: torch.tensor,
            text_prev_lengths: torch.tensor,
            text_ctc: torch.tensor,
            text_ctc_lengths: torch.tensor,
            task: int = None
    ):
        """
            Consolidates one batch of utterances

            :param torch.tensor speech: size (Batch_size, Time_steps, Dimension)
            :param torch.tensor speech_lengths:
            :param torch.tensor text:
            :param torch.tensor text_lengths:
            :param int task:
        """
        pass

    """
        Save the "consolidated knowledge"
    """
    @abstractmethod
    def save(
            self,
    ):
        pass


class Regularization(Consolidation):
    def __init__(
            self,
            model: torch.nn.Module,
            device: str,
            outdir: str,
            name: str,
            exclude: typing.List[str] = None,
            prev_outdir: str = "",
            max_samples: int = -1,
            ignore_old: bool = False,  # if True, old importance weights are ignored - even if they exist
    ):
        """
            Regularization Abstract Class

            Generic class to be inherited by Regularization-based CL methods (EWC, MAS, etc.) for Consolidation

            :param torch.nn.Module model: the model to consolidate knowledge from
            :param str device
            :param str outdir: directory to store the importance weights
            :param str name: name of the method
            :param List[str] exclude: list of layers to exclude from regularization
            :param str prev_outdir: output directory of pretrained model to add importance weights
        """
        self.include = lambda name: exclude is None or not name in exclude
        logging.info("Following layers excluded from regularization: %s" % str(exclude))
        self.device = device
        name = 'importance_weight.%s' % name
        self.name = name
        self.outdir = outdir
        self.save_name = "%s/%s" % (outdir, name)
        self.model = model.eval().to(device)
        self.importance_weights = {name: torch.zeros_like(param).cpu()
                                   for name, param in model.named_parameters() if self.include(name)}
        self.samples = 0
        self.max_samples = max_samples
        if prev_outdir and not ignore_old:
            prev_outdir = '/'.join(prev_outdir.split("/")[:-1])
            if os.path.isfile("%s/%s" % (prev_outdir, name)):
                self.old_importance_weights = torch.load("%s/%s" % (prev_outdir, name))
                logging.info("Found importance weights from previous tasks: %s/%s" % (prev_outdir, name))
            else:
                logging.info("WARNING: This is not the first task, but no old importance weights were found!")
                logging.warning("WARNING: This is not the first task, but no old importance weights were found!")
                self.old_importance_weights = {n: torch.zeros_like(p) for n, p in self.importance_weights.items()}
        else:
            self.old_importance_weights = {n: torch.zeros_like(p) for n, p in self.importance_weights.items()}

    def stop(self):
        return 0 < self.max_samples <= self.samples

    """
        Save the importance weights
    """
    def save(
            self,
    ):
        # divide importance weights by number of seen utterances
        self.importance_weights = {name: param / self.samples for name, param in self.importance_weights.items()}
        # add old importance weights
        self.importance_weights = {n: p + self.old_importance_weights[n] for n, p in self.importance_weights.items()}
        torch.save(self.importance_weights, self.save_name)


"""  
########################### REGULARIZATION-BASED CL METHODS ##########################################################
"""

class Kronecker(Regularization):
    def __init__(
            self,
            model: torch.nn.Module,
            device: str,
            outdir: str,
            exclude: typing.List[str] = None,
            prev_outdir: str = "",
            max_samples: int = -1,
            name: str = "kf"
    ):
        super(Kronecker, self).__init__(model, device, outdir, name, exclude, prev_outdir, max_samples)
        if prev_outdir:
            prev_outdir = '/'.join(prev_outdir.split("/")[:-1])
            if os.path.isfile("%s/%s" % (prev_outdir, name)):
                self.old_kf = torch.load("%s/%s" % (prev_outdir, name))
                logging.info("Found KF weights from previous tasks: %s/%s" % (prev_outdir, name))
            else:
                self.old_kf = None
        else:
            self.old_kf = None
        self._prepare_model(self.model)
        self.Q, self.H = {}, {}

    def _prepare_model(
            self,
            model: torch.nn.Module,
    ):
        # new AdaptedLinear module
        class MemoryLinear(torch.nn.Linear):
            def __init__(
                    self,
                    in_features: int,
                    out_features: int,
                    **kwargs
            ):
                torch.nn.Linear.__init__(self, in_features, out_features, **kwargs)
                self.input = None
                self.output = None
                self.backprop = None
                self._retain_output_grad = False  # set True if you specifically want out.grad

            def forward(self, x: torch.Tensor):
                # Cache activations for curvature stats; DETACH to avoid keeping the whole graph
                self.input = x.detach()
                out = F.linear(x, self.weight, bias=None)
                self.output = out
                # If you really want out.grad later (non-leaf), you must retain it
                if self._retain_output_grad:
                    out.retain_grad()
                # Register a hook to capture the backward signal (grad_output)
                # grad_output is a tuple; grad_output[0] has shape like out
                out.register_hook(self._save_backprop)
                # 4) Now add the bias (this does not change dL/dz)
                if self.bias is not None:
                    # works for any input rank; adds along last dim
                    return out + self.bias
                    # equivalently: out = z + self.bias
                else:
                    return out

            def _save_backprop(self, grad_output: torch.Tensor):
                # Detach to avoid autograd tracking during your EMA/stat updates
                self.backprop = grad_output.detach()

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and self.include(name):
                # create new module
                new_module = MemoryLinear(module.in_features, module.out_features, bias=module.bias is not None)
                parent_module, target_name, target_module = get_submodules(model, name)
                replace_module(parent_module, target_name, module, new_module, copy_weight=True)

    def consolidate(
            self,
            speech: torch.tensor,
            speech_lengths: torch.tensor,
            text: torch.tensor,
            text_lengths: torch.tensor,
            text_ctc: torch.tensor,
            text_ctc_lengths: torch.tensor,
            text_prev: torch.tensor,
            text_prev_lengths: torch.tensor,
            task: int = None
    ):
        if 0 < self.max_samples <= self.samples:
            return
        cuda_overview('start of consolidate')
        # set the batch size: if speech has only two dimensions, it is 1
        batch_size = speech.size(0)
        # zero the gradients of the model
        self.model.zero_grad()
        # compute the loss and gradients
        inputs = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "text_prev": text_prev,
            "text_prev_lengths": text_prev_lengths,
            "text_ctc": text_ctc,
            "text_ctc_lengths": text_ctc_lengths,
        }
        # move to device
        inputs = _recursive_to(inputs, self.device)
        # compute the loss and gradients
        loss = self.model(**inputs)
        loss.backward()
        # store squared gradients
        self._process_linear_layers(self.model, loss)
        # keep track of the batch size
        self.samples += batch_size
        logging.info("KF has consolidated %d utterances.." % self.samples)

    def _process_linear_layers(
            self,
            model: torch.nn.Module,
            loss,
    ):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and self.include(name):
                Q, H = self._compute_q_and_h(module, loss)
                if not name in self.Q.keys():
                    self.Q[name], self.H[name] = Q, H
                else:
                    self.Q[name] += Q
                    self.H[name] += H

    def _compute_q_and_h(
            self,
            module: torch.nn.Module,
            loss,
    ):
        # compute Q from input
        p = module.input.view(-1, module.input.size(2))  # (B, L, d) --> (B * L, d)
        Q = (1 / p.size(0)) * torch.matmul(torch.transpose(p, 0, 1), p)
        # compute H from pre-activations
        g = module.backprop
        #logging.info(f"g = {g.size()}")
        g = module.backprop.reshape(-1, module.backprop.size(2))  # (B, L, d) --> (B * L, d)
        H = (1 / g.size(0)) * torch.matmul(torch.transpose(g, 0, 1), g)
        return Q.detach(), H.detach()

    def save(
            self,
    ):
        for name in self.Q.keys():
            self.Q[name] = self.Q[name] / self.samples
            self.H[name] = self.H[name] / self.samples
        new_kf = {}
        for name in self.Q.keys():
            new_Q = [self.Q[name]]
            new_H = [self.H[name]]
            new_kf[name] = {'Q': new_Q, 'H': new_H}
        torch.save(new_kf, self.save_name)

    def stop(self):
        return 0 < self.max_samples <= self.samples
