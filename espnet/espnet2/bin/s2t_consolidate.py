#!/usr/bin/env python3
import os
try:
    print("CUDA_VISIBLE_DEVICES = %s" % str(os.environ['CUDA_VISIBLE_DEVICES']))
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].replace("CUDA", "")
    print("CUDA_VISIBLE_DEVICES = %s" % str(os.environ['CUDA_VISIBLE_DEVICES']))
except KeyError as ke:
    print("WARNING: Could not set CUDA_VISIBLE_ERROR due to error (%s)" % ke)



import argparse
import logging
from pathlib import Path
import sys
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List
import yaml
import inspect


import espnet2.legacy.nets.pytorch_backend as c2

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.s2t import S2TTask
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.train.class_choices import ClassChoices


def yaml2args(config_file=""):
    if not config_file:
        return {}
    config_file = Path(config_file)
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)
    return vars(args)



cl_choices = ClassChoices(
    "cl_method",
    classes=dict(
        kf=c2.Kronecker,
    ),
    type_check=c2.Consolidation,
    optional=False,
)



class Speech2Task:
    """ Speech2Task class """

    def __init__(
        self,
        output_dir: str = None,
        s2t_train_config: Union[Path, str] = None,
        s2t_model_file: Union[Path, str] = None,
        cl_method: str = "ewc",
        device: str = "cpu", 
        batch_size: int = 1,
        dtype: str = "float32",
        ex_layers: str = "",
        config_file: Union[Path, str] = None,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        logging.info(f"s2t_model_file = {s2t_model_file}")
        s2t_model, s2t_train_args = S2TTask.build_model_from_file(
            s2t_train_config, s2t_model_file, device, use_adapter=False,
        )
        s2t_model.to(dtype=getattr(torch, dtype)).eval()
        self.s2t_train_args = s2t_train_args
        self.s2t_model = s2t_model.to(device)
        # 2. Set some params
        self.device = device
        logging.info("Device = %s" % (device))
        self.dtype = dtype        
        self.outdir = output_dir
        self.init_model = output_dir + '/initial_model.pth'
        
        # 3. Prepare CL method
        cl_method_class = cl_choices.get_class(cl_method)
        
        # 4. Get the arguments based on __init__()
        # we write a function to this end
        def get_args(args, method, conf):
            # inspect is needed to 'inspect' __init__()
            opt_args = {
                    'model': self.s2t_model, 
                    'init_model': self.init_model,
                    'device': self.device,
                    'outdir': self.outdir,
                    'ctc_weight': s2t_train_args.model_conf['ctc_weight'],
                    'input_size': s2t_train_args.encoder_conf['output_size'],
                    'prev_outdir': s2t_train_args.init_param[0] if len(s2t_train_args.init_param) > 0 else "",
                       }
            if not 'exclude' in conf.keys():
                opt_args['exclude'] = self.exclude(self.s2t_model, ex_layers)
            cons_args = inspect.getfullargspec(method.__init__).args
            return {arg: val for arg, val in opt_args.items()
                    if arg in cons_args}

        # read the arguments from .yaml file
        primary_args = yaml2args(config_file)
        # add extra arguments based on __init__
        extra_args = get_args(self, cl_method_class, primary_args)
        
        logging.info("config_file = %s" % config_file)

        # 5. Set the Consolidate object
        cons_args = {**primary_args, **extra_args}
        self.cl_method = cl_method_class(**cons_args) 

        # 6. if necessary, set max number of to-be-processed utts
        if hasattr(self.cl_method, 'max_samples'):
            self.max_samples = self.cl_method.max_samples
        else:
            self.max_samples = -1

    def stop(
            self,
            processed_utts: int,
    ):
        return 0 <= self.max_samples <= processed_utts

    def exclude(
            self, 
            model: torch.nn.Module,
            strategy: str,
        ):
        ex_strategy = {'task-specific': lambda name: 'prompt' in name or 'adapter' in name,
                       'prompt': lambda name: 'prompt' in name,
                       'adapter': lambda name: 'adapter' in name,}
        exclude_layers = []
        if strategy in ex_strategy.keys():
            logging.info("--> well-known exclude_strategy")
            exclude = ex_strategy[strategy]
            for k, p in model.named_parameters():
                if exclude(k):
                    logging.info(f"Setting {k}.regularize = False")
                    exclude_layers.append(k)
        return exclude_layers

    def __call__(
            self,
            speech: torch.tensor,
            speech_lengths: torch.tensor,
            text: torch.tensor,
            text_lengths: torch.tensor,
            text_ctc: torch.tensor,
            text_ctc_lengths: torch.tensor,
            text_prev: torch.tensor,
            text_prev_lengths: torch.tensor,
            task: int = None,
        ):

        self.cl_method.consolidate(
                speech=speech, 
                speech_lengths=speech_lengths, 
                text=text, 
                text_lengths=text_lengths, 
                text_prev=text_prev,
                text_prev_lengths=text_prev_lengths,
                text_ctc=text_ctc,
                text_ctc_lengths=text_ctc_lengths,
        )



    def save(self):
        self.cl_method.save()


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    train_data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    allow_variable_data_keys: bool,
    s2t_train_config: Optional[str],
    s2t_model_file: Optional[str],
    cl_method: str = "ewc", 
    exclude: str = "",
    task: int = None,
    config_file: Optional[str] = "",
    log_level=logging.WARNING,
    **kwargs,
):

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    logging.info("ngpu = %d, device = %s" % (ngpu, device))

    # 1. Set random-seed
    set_all_random_seed(seed)

    logging.info("Building Speech2Task model..")

    # 2. Build speech2task model
    speech2task = Speech2Task(
            output_dir=output_dir,
            s2t_train_config=s2t_train_config,
            s2t_model_file=s2t_model_file,
            device=device,
            batch_size=batch_size,
            dtype=dtype,
            ex_layers=exclude,
            cl_method=cl_method,
            config_file=config_file,
    )

    logging.info("Building iterators with %d workers..." % num_workers)
 
    # 3. Build data-iterator
    trainloader = S2TTask.build_streaming_iterator(
        train_data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=S2TTask.build_preprocess_fn(speech2task.s2t_train_args, False),
        collate_fn=S2TTask.build_collate_fn(speech2task.s2t_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=False,
    )

    # 4 .Start for-loop for training
    it, processed_utts  = 1, 0
    for keys, batch in trainloader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        # call speech2task
        speech2task(**batch)
        # update iter number and processed_utts
        it += 1
        processed_utts += batch['speech'].size(0)
        if speech2task.stop(processed_utts):
            logging.info("Stopping consolidation: Processed %d utterances.." % processed_utts)
            break

    speech2task.save()

        
def get_parser():
    parser = config_argparse.ArgumentParser(
        description="S2T Consolidation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--train_data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--s2t_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--s2t_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--cl_method",
        type=str,
        help="Continual Learning method",
        default="ewc",
    )
    group.add_argument(
        "--exclude",
        type=str,
        help="Strategy to exclude layers from regularization",
        default="",
    )

    group.add_argument(
        "--config_file",
        type=str,
        help="Consolidation configuration",
    )


    group.add_argument(
        "--lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--ngram_file",
        type=str,
        help="N-gram parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths."
        "If maxlenratio<0.0, its absolute value is interpreted"
        "as a constant max output length",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)
    group.add_argument("--train", type=str2bool, default=True)
    group.add_argument(
        "--transducer_conf",
        default=None,
        help="The keyword arguments for transducer beam search.",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--task",
        type=int,
        default=None,
        help="Task id of current task - required for task-specific layers")
    group.add_argument(
        "--lang",
        type=str,
        default="",
        help="Task name of current task")

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
