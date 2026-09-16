# Copyright 2022 Steven Vander Eeckt

""" Continual Learning extensions for End-to-End ASR (ESPnet2) """
import sys
import time
import numpy as np
import torch
import logging
from abc import ABC, abstractmethod
import os
import torch.nn.functional as func
import copy
import typing
import random

import espnet2.layers.loralib as lora


from espnet2.torch_utils.model_summary import model_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################################################################################################
########################### ABSTRACT CLASSES ########################################################################
#####################################################################################################################


class CLMethod(ABC):
    """
        CL Method Abstract Class
    """
    pass


class FineTuningLinear(CLMethod):
    def __init__(self, device: str):
        pass

    def set_model(self, model: torch.nn.Module):
        """
        Freeze all non-Linear modules.

        For Linear-like modules:
        - keep existing requires_grad values unchanged
        - freeze biases
        """
        linear_param_ids = set()
        bias_param_ids = set()

        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                for name, p in m.named_parameters(recurse=False):
                    linear_param_ids.add(id(p))
                    if name == "bias":
                        bias_param_ids.add(id(p))

        for name, p in model.named_parameters():
            old_requires_grad = p.requires_grad

            if id(p) in bias_param_ids:
                # Linear bias: always frozen.
                p.requires_grad = False
            elif id(p) in linear_param_ids:
                # Linear weight or other direct Linear parameter:
                # leave unchanged.
                p.requires_grad = old_requires_grad
            else:
                # Non-Linear parameter: frozen.
                p.requires_grad = False

            if old_requires_grad and not p.requires_grad:
                logging.info(
                    f"Freezing {name} with {p.data.numel()} parameters."
                )

        logging.info(model_summary(model))
