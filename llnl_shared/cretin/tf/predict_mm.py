# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to run Cretin model
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import yaml
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from modelzoo.common.tf.estimator.cs_estimator import CerebrasEstimator
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.run_utils import (
    check_env,
    get_csrunconfig_dict,
    is_cs,
    save_params,
    update_params_from_args,
)
from llnl_shared.cretin.tf.data import input_fn
from llnl_shared.cretin.tf.model import model_fn
from llnl_shared.cretin.tf.utils import get_params

from cerebras.tf.model_fusion import MultiModelFusion
from cerebras.tf.model_fusion import ModelInfo
from cerebras.tf.model_fusion import MULTI_MODEL_STACK_KEY
from cerebras.pb.stack.full_pb2 import FullConfig

def rebatch_input_fn(input_fn, batch_size=1):
    """
    Returns a modified input_fn with the batch size set to `batch_size`.
    :param input_fn: The original input_fn.
    :param batch_size: Batch size to set for the new input_fn. Defaults to 1.
    :returns: A modified input_fn whose batch size is `batch_size`.
    """
    def _rebatched_input_fn(params):
        ds = input_fn(params)
        ds = ds.unbatch()
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds
    return _rebatched_input_fn

def build_cretin_instance(
    params_path,
    deltat_scale_factor: float,
    scenario: str = "base",
    checkpoint_dir: Optional[str] = None,
    tfrecord_path: Optional[str] = None,
    n_towers: Optional[int] = 1,
    ):
    """
    Returns a CRETIN model info.
    :param params_path: path to yaml file with the model config (e.g. 'params.yaml')
    :param deltat_scale_factor: DeltaT scaling factor to use.
    :param scenario: CRETIN scenario to build.
    :param checkpoint_dir: directory that contains checkpoint to use for preloading weights
    :param tfrecord_path: path to tfrecord to use for this cretin instance
    :param n_towers: Number of identical towers to place in parallel for this instance 
    :returns: A model info object.
    """
    params = get_params(params_path, config=scenario)
    params["runconfig"]["mode"] = "infer"
    params['model']['n_towers'] = n_towers
    if n_towers > 1:
        params['model']['in_parallel'] = True
    if tfrecord_path:
        params['inference']['infer_input'] = tfrecord_path
    return ModelInfo(
        model_fn=model_fn,
        input_fn=rebatch_input_fn(input_fn),
        params=params,
        deltat_scaling_factor=deltat_scale_factor,
        name=f"model_{scenario}",
        checkpoint=checkpoint_dir,
    )

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="run multi-model inference. Use the compile_only or \
                validate_only flag to stop after compilation. \
                example: python predict_mm.py -c params_mm.yaml"
    )
    parser.add_argument(
        "--cs_ip", help="CS-1 IP address, defaults to None", default=None
    )
    parser.add_argument(
        "-p",
        "--params",
        help="path to params yaml file",
        required=False,
        default="./params.yaml",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=[
            "save_graph_only",
            "infer",
            "infer_cpp",
            "compile_only",
            "validate_only",
        ],
        help=(
            "Valid modes are save_graph_only, infer, infer_cpp. save_graph_only is used /"
            "for saving graph to events file for visualization"
        ),
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        type=str,
        help="Save compilation and non-simfab outputs",
        default="./model_dir",
    )
    parser.add_argument(
        "-c",
        "--multimodel_params",
        required=False,
        default="./params_mm.yaml",
        help="path to multimodel params yaml file that contains information /"
            "for each variant. /" 
            "The model variants should be specified as a list of entries. See /"
            "params_mm.yaml as an example."
    )
    parser.add_argument(
        "--steps",
        required=False,
        help="prediction uses number of samples which is number of inference /"
              "requests. Currently it is fetched from runconfig params"
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Compile model up to kernel matching.",
        default=False,
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        help="Compile model completely, generating compiled executables.",
        default=False,
    )
    
    return parser

def main():
    # SET UP
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    params_path = args.params
    params = get_params(params_path)
    with open(args.multimodel_params,'r') as fid:
        mm_params = yaml.load(fid)
    models = []
    for variant in mm_params:
        scenario = variant['scenario']
        count = variant['count']
        n_towers = variant.get('n_towers',1)
        models.extend([
            build_cretin_instance(
                params_path,
                deltat_scale_factor=1,
                scenario=scenario,
                n_towers=n_towers,
            )
            for _ in range(int(count))
        ])
    fusion = MultiModelFusion(models)

    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    # save params for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])
    # get runtime configurations
    use_cs = is_cs(runconfig_params)
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    check_env(runconfig_params)
    if params["runconfig"]["mode"] == "infer_cpp":
        # This import will result in global imports of modules that are built
        # and thus not accessible on a gpu run (will result in import error).
        # So moving the import to the context it is needed.
        from cerebras.tf.utils import prep_orchestrator
        prep_orchestrator()
    stack_params = None
    if use_cs:
        stack_params = {
            MULTI_MODEL_STACK_KEY: fusion.stack_params,
        }
    config = FullConfig()
    config.matching.add_pack_and_unpack.max_egress_per_pack = 1
    config.placement.prep_recolor_kernels.wrap_pack_kernel = True
    stack_params["config"] = config
    est_config = CSRunConfig(
        stack_params=stack_params,
        cs_ip=runconfig_params["cs_ip"],
        **csrunconfig_dict,
    )
    est = CerebrasEstimator(
        model_fn=fusion.model_fn,
        model_dir=runconfig_params["model_dir"],
        config=est_config,
        params=fusion.params,
    )
    output = None
    if args.compile_only or args.validate_only:
        est.compile(
            fusion.input_fn,
            validate_only=args.validate_only,
        )
    elif params["runconfig"]["mode"] == 'save_graph_only':
        # call model_fn just to generate events file with graph
        with tf.compat.v1.Graph().as_default():
            inp = {}
            for ii in range(len(fusion.params)):
                features = {
                    'input': tf.compat.v1.placeholder(
                        tf.float16,
                        [1,params['train_input']['input_len']],
                    ),
                    'input_params': tf.compat.v1.placeholder(tf.float16, [1,2]),
                }
                inp['model_%d'%ii] = features
            # Creates the graph
            _ = fusion.model_fn(inp, None, 'infer', fusion.params)
            # Export the graph to output directory
            with tf.compat.v1.Session() as sess:
                tf.compat.v1.summary.FileWriter(args.model_dir, sess.graph)
    elif params["runconfig"]["mode"] == tf.estimator.ModeKeys.PREDICT:
        pred_dir = os.path.join(runconfig_params["model_dir"], "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        sys_name = "cs" if use_cs else "tf"
        file_to_save = f"predictions_{sys_name}_{est_config.task_id}.npz"
        output = []
        num_samples = runconfig_params["infer_steps"]
        preds = est.predict(
            input_fn=fusion.input_fn, num_samples=num_samples, use_cs=use_cs
        )
        for pred in preds:
            output.append(pred)
        if len(output) > 0:
            np.savez(os.path.join(pred_dir, file_to_save), output)
    elif params["runconfig"]["mode"] == "infer_cpp":
        preds = est.predict(
            input_fn=fusion.input_fn, num_samples=1, use_cs=True
        )
        

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
