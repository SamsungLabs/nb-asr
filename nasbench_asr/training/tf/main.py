# pylint: skip-file
# import argparse
import logging
import os
import sys
import random
import numpy as np
from attrdict import AttrDict
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
os.environ['TF_DETERMINISTIC_OPS'] = '1'


from nasbench_asr.quiet_tensorflow import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True)

logging.getLogger("tensorflow").setLevel(logging.FATAL)
from .core.trainer.callbacks.model_checkpoint_max_to_keep import ModelCheckpointMaxToKeep
from .core.trainer.callbacks import lrscheduler 
from .core.trainer.callbacks.nni_report import NNIReport
from .core.trainer.callbacks.tensorboard_image import Tensorboard
from .core.utils import (expand_path_if_nni, read_yaml,
                        update_config_with_nni_tuned_params)

from .datasets.datasets import get_data
from .core.learner.asr_ctc import ASRCTCLearner

from ...model.tf.model import *
from ...model.tf.ops import *

try:
    import nni
except ImportError:
    print("Microsoft NNI is not installed, and will therefore not be used")

def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    
    # hvd_util.init_gpus()
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass

    verbose = config.verbose #if hvd.rank() == 0 else 0

    data_train, data_validate, data_test = get_data(config.dataset)

    model_configs = gen_model_configs(shuffle=False)
    model_config = model_configs[config.counter]
    #model_config = [[3,1], [4,1,1], [0,0,0,1]]
    model = ASRModel(model_config=config, sc_config=model_config, num_classes=data_train.encoder.encoder.vocab_size, use_rnn=config.use_rnn,
                    **{
                           "input_shape": [None, config.dataset.featurizer.num_feature_filters],
                           "epsilon": 0.001,
                           "stats": data_train.stats,
                           "mask_time_fraction": 0.0,
                           "mask_channels_fraction": 0.0,
                           "seed": config.seed
                    }
                    )

    learning_rate = config.train.lr #* hvd.size()
    optimizer = getattr(tf.keras.optimizers, config.train.optimizer)(learning_rate)

    # Adding learning rate scheduler callback
    lr_scheduler = getattr(lrscheduler, config.lrscheduler.name).from_dict(config)
    callbacks = [lr_scheduler]

    ckpt_dir = expand_path_if_nni(config.callbacks.model_checkpoint.log_dir)
    # if hvd.rank() == 0:
    if config.callbacks.model_checkpoint.max_to_keep >= 1:
        # use cb ModelCheckpoint before cb NNIReport so that the weights saved
        # by the former get copied by the latter at the end of each epoch
        callbacks.append(
            ModelCheckpointMaxToKeep(
                folder=ckpt_dir,
                monitor=config.callbacks.monitor,
                mode=config.callbacks.mode,
                max_to_keep=config.callbacks.model_checkpoint.max_to_keep,
            ))
    callbacks.append(
        NNIReport(nni_chosen_metric=config.callbacks.nni.nni_chosen_metric,
        report_final_result=False))
    tensorboard_dir = expand_path_if_nni(config.callbacks.tb.log_dir)
    callbacks.append(Tensorboard(log_dir=tensorboard_dir, update_freq=10))
    # profile_batch=0 required on mlp to show train on TensorBoard
    # https://stackoverflow.com/questions/58300465/tensorboard-2-0-0-not-updating-train-scalars
    # callbacks.append(
    #    tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir,
    #                                   update_freq=10,
    #                                   profile_batch=0))

    asr_ctc_learner = ASRCTCLearner(
        model=model,
        optimizer=optimizer,
        get_decoded_from_encoded=data_train.encoder.get_decoded_from_encoded,
        fp16_allreduce=config.train.fp16_allreduce,
        greedy_decoder=config.test.greedy_decoder,
        beam_width=config.test.beam_width,
        encoder_class=config.dataset.encoder.encoder_class)

    asr_ctc_learner.compile()
    asr_ctc_learner.model._model.summary()

    history_fit = asr_ctc_learner.fit(
        data_train.ds,
        epochs=config.train.epochs,
        steps_per_epoch=data_train.steps,
        callbacks=callbacks,
        validation_data=data_validate.ds,
        validation_steps=data_validate.steps,
        verbose=verbose,
    )

    # if hvd.rank() == 0:
    tf.print(history_fit.history)
    with open("./history_fit.pickle", "wb") as fp:
        pickle.dump(history_fit.history, fp)

    if config.callbacks.model_checkpoint.max_to_keep >= 1:
        # load best model seen so far for the test eval
        lers = history_fit.history[config.callbacks.monitor]
        best_model_epoch = lers.index(min(lers)) 
        checkpoint_path = ckpt_dir + '/cp-' + str(best_model_epoch).zfill(4) + '.ckpt'
        asr_ctc_learner.model.load_weights(checkpoint_path)
    
    tmp = asr_ctc_learner.evaluate(data_test.ds,
            verbose=verbose,
            steps=data_test.steps,
            return_dict=True)

    # if hvd.rank() == 0:
    history_evaluate = {}
    for key, val in tmp.items():
        history_evaluate['val_'+key] = val
        if key == config.callbacks.nni.nni_chosen_metric:
            history_evaluate["default"] = val

    if "nni" in sys.modules:
        nni.report_intermediate_result(history_evaluate)
        nni.report_final_result(history_evaluate['default'])
    tf.print(history_evaluate)
    with open("./history_evaluate.pickle", "wb") as fp:
        pickle.dump(history_evaluate, fp)

    os.removedirs(ckpt_dir)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-f', '--config_file',
    #     type=str,
    #     required=True,
    #     help='Input configuration file name.'
    # )
    # cmd_data, _ = parser.parse_known_args()
    # config = read_yaml(cmd_data.config_file)
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    config = read_yaml(os.path.join(curr_dir, 'config.yaml'))
    config.dataset = read_yaml(os.path.join(curr_dir, config.dataset.config))

    if "nni" in sys.modules:
        tuned_params = nni.get_next_parameter()
        config = update_config_with_nni_tuned_params(config, tuned_params)

    main(config)
