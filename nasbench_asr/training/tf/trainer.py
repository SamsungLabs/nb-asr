import pickle
import pathlib
import functools
import collections.abc as cabc

import numpy as np
from nasbench_asr.quiet_tensorflow import tensorflow as tf

from .callbacks.tensorboard import Tensorboard
from .callbacks.lrscheduler import ExponentialDecay
from .callbacks.reset_states import ResetStatesCallback
from .metrics.ratio import Ratio
from .metrics.ler import get_ler_numerator_denominator
from .metrics.wer import get_wer_numerator_denominator
from .metrics.ctc import get_logits_encodeds, get_normalized_ctc_loss_without_reduce
from .datasets.timit_foldings import old_to_new_indices, get_phoneme_mapping


def get_logits_size(features, features_size, logits):
    time_reduction = tf.cast(tf.shape(features)[1],
                             dtype=tf.float32) / tf.cast(tf.shape(logits)[1],
                                                         dtype=tf.float32)
    logits_size = tf.cast(tf.cast(features_size, dtype=tf.float32) /
                          time_reduction,
                          dtype=features_size.dtype)

    return logits_size


def get_loss():
    def loss(logits, logits_size, encodeds, encodeds_size, metrics=None):
        logits_transposed = tf.transpose(logits, [1, 0, 2])
        ctc_loss_without_reduce = get_normalized_ctc_loss_without_reduce(
            logits_transposed=logits_transposed,
            logits_size=logits_size,
            encodeds=encodeds,
            encodeds_size=encodeds_size,
        )

        ctc_loss_without_reduce_numerator = ctc_loss_without_reduce
        ctc_loss_without_reduce_denominator = tf.ones_like(ctc_loss_without_reduce)

        if metrics is not None:
            metrics.update({
                "ctc_loss": (
                    ctc_loss_without_reduce_numerator,
                    ctc_loss_without_reduce_denominator,
                )
            })

        return tf.reduce_mean(ctc_loss_without_reduce)

    return loss


class Trainer():
    class RememberBestCallback(tf.keras.callbacks.Callback):
        def __init__(self, trainer, checkpoint_name):
            self.trainer = trainer
            self.best_so_far = None
            self.checkpoint_name = checkpoint_name

        def on_epoch_end(self, epoch, logs=None):
            if 'val_ler' in logs:
                value = logs['val_ler']
                if self.best_so_far is None or value <= self.best_so_far:
                    self.best_so_far = value
                    self.trainer.remember_best()
                    self.trainer.save(self.checkpoint_name)
            else:
                print('Missing validation LER')

    class SaveLatestCallback(tf.keras.callbacks.Callback):
        def __init__(self, trainer, checkpoint_name):
            self.trainer = trainer
            self.checkpoint_name = checkpoint_name

        def on_epoch_end(self, epoch, logs=None):
            self.trainer.save(self.checkpoint_name)

    def __init__(self, dataloaders, loss, gpus=None, save_dir=None, verbose=True):
        encoder, data_train, data_validate, data_test = dataloaders

        self.encoder = encoder
        self.data_train = data_train
        self.data_validate = data_validate
        self.data_test = data_test

        self.save_dir = save_dir
        if self.save_dir:
            pathlib.Path(self.save_dir).mkdir(exist_ok=True)
        self.verbose = verbose

        self.model = None
        self.optimizer = None
        self.trackers = {}
        self.loss = loss

        self.get_decoded_from_encoded = self.data_train.encoder.get_decoded_from_encoded
        self.fp16_allreduce = True
        self.greedy_decoder = False
        self.beam_width = 12

        self._best_weights = None

        if gpus is not None and (not isinstance(gpus, cabc.Sequence) or bool(gpus)):
            if not isinstance(gpus, cabc.Sequence):
                gpus = [gpus]
        else:
            gpus = []

        if len(gpus) != 1:
            raise ValueError('TF implementation only supports running on a single GPU')

    #
    # API
    #

    def train(self, model, epochs=40, lr=0.0001, reset=False, model_name=None):
        metrics = {
            "ctc_loss": Ratio,
            "wer": Ratio,
            "ler": Ratio
        }
        self.init_trackers(metrics)

        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(lr)

        # Adding learning rate scheduler callback
        self.lr_scheduler = ExponentialDecay(0.9, start_epoch=5, min_lr=0.0, verbose=False)

        callbacks = [self.lr_scheduler]

        this_model_save_dir = None
        if self.save_dir is not None:
            this_model_save_dir = self.save_dir
            if model_name is not None:
                this_model_save_dir = this_model_save_dir / model_name

            latest_ckpt = this_model_save_dir / 'latest.ckpt'
            best_ckpt = this_model_save_dir / 'best.ckpt'
            tensorboard_dir = this_model_save_dir / 'tensorboard'
            callbacks.append(Trainer.SaveLatestCallback(self, latest_ckpt))
            callbacks.append(Trainer.RememberBestCallback(self, best_ckpt))
            callbacks.append(Tensorboard(log_dir=tensorboard_dir, update_freq=10))

            # TODO: is that enough to restore state? or maybe we should always start from the beginnin
            if best_ckpt.exists():
                if reset:
                    best_ckpt.unlink()
                else:
                    self.load(best_ckpt)
                    self.remember_best()
            if latest_ckpt.exists():
                if reset:
                    latest_ckpt.unlink()
                else:
                    self.load(latest_ckpt)

        self.compile()
        if self.verbose:
            self.model._model.summary()

        history_fit = self.fit(
            self.data_train.ds,
            epochs=epochs,
            steps_per_epoch=self.data_train.steps,
            callbacks=callbacks,
            validation_data=self.data_validate.ds,
            validation_steps=self.data_validate.steps,
            verbose=self.verbose
        )
        if self.verbose:
            tf.print(history_fit.history)

        self.recall_best()

        test_res = self.evaluate(self.data_test.ds,
                verbose=self.verbose,
                steps=self.data_test.steps,
                return_dict=True)

        history_evaluate = {}
        for key, val in test_res.items():
            history_evaluate['val_'+key] = val
        if self.verbose:
            tf.print(history_evaluate)

        if self.save_dir:
            with open(this_model_save_dir / 'scores.pickle', "wb") as fp:
                pickle.dump(history_fit.history, fp)
            with open(this_model_save_dir / 'test_scores.pickle', "wb") as fp:
                pickle.dump(history_evaluate, fp)

        self.model = None
        self.optimizer = None
        self.trackers = {}

    def step(self, input, training=True):
        if training:
            return self._train_step(input)
        else:
            return self._test_step(input)

    def save(self, checkpoint):
        self.model.save_weights(filepath=checkpoint, overwrite=True, save_format='tf')

    def load(self, checkpoint):
        self.model.load_weights(checkpoint)

    def remember_best(self):
        self._best_weights = self.model.get_weights()

    def recall_best(self):
        self.model.set_weights(self._best_weights)

    #
    # Implementation
    #

    def init_trackers(self, metrics, track_modes=["train", "test"]):
        """
        Initializing metric-trackers for e.g., training and validation datasets
        Args:
            metrics : A dictionary containing the tracker callable classes as values
            track_mode : A list containing the dataset partitions, e.g., ["train", "val", "test"]
        """
        self.trackers.clear()

        if metrics is None:
            return

        assert isinstance(metrics, dict), "Metrics should be a dictionary with callable values"
        assert isinstance(track_modes, list), "Expecting a list of tracking modes, e.g., [train, test]"

        for mode in track_modes:
            self.trackers[mode] = {}

            for metric_name, metric_fn in metrics.items():
                if not callable(metric_fn): continue

                name = mode + "_" + metric_name
                self.trackers[mode][metric_name] = metric_fn(name=name)
                self.trackers[mode][metric_name].reset_states()

    def get_tracker_results(self):
        """
        Prints all tracker result on screen
        """
        results = {}

        # Looping over all trackers
        for key in self.trackers.keys():
            metrics = self.trackers.get(key)

            for met in metrics.keys():
                val = metrics.get(met).result()

                # Only printing tracker results if not NaN
                if not tf.math.is_nan(val):
                    results[f"{key}-{met}"] = val.numpy()

        return results

    def _train_step(self, data):
        """
        A wrapper to update the train trackers
        """
        logs = self.train_step(data)

        for key in logs.keys() & self.trackers["train"].keys():
            self.trackers["train"][key].update_state(logs[key])
            logs[key] = self.trackers["train"][key].result()

        # the returned logs will appear in the arguments to the methods of
        # the classes inheriting from tf.keras.callbacks.Callback

        return logs

    def _test_step(self, data):
        """
        A wrapper to update the test trackers
        """
        logs = self.test_step(data)

        for key in logs.keys() & self.trackers["test"].keys():
            self.trackers["test"][key].update_state(logs[key])
            logs[key] = self.trackers["test"][key].result()

        # the returned logs will appear in the arguments to the methods of
        # the classes inheriting from tf.keras.callbacks.Callback

        return logs

    def compile(self):
        """
        Overrides the tf.keras.Model train_step/test_step functions and
        compiles the model
        """
        self.model.train_step = functools.partial(self.step, training=True)
        self.model.test_step = functools.partial(self.step, training=False)

        self.model.compile(optimizer=self.optimizer)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):

        """
        Wrapper for
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        method of self.model, which ensures that HorovodCallback is prepended
        to callbacks before calling built-in fit. Parameters and default
        values are the same as those in built-in fit.
        """

        # Adding ResetStateCallback as default
        if callbacks is None:
            callbacks = []
        callbacks.insert(0, ResetStatesCallback(self.trackers))

        return self.model.fit(x=x,
                              y=y,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=self.verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data,
                              shuffle=shuffle,
                              class_weight=class_weight,
                              sample_weight=sample_weight,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_batch_size=validation_batch_size,
                              validation_freq=validation_freq,
                              max_queue_size=max_queue_size,
                              workers=workers,
                              use_multiprocessing=use_multiprocessing)


    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False,
                 return_dict=False):

        """
        Wrapper for
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
        method of self.model, which ensures that HorovodCallback is prepended
        to callbacks before calling built-in evaluate. Parameters and default
        values are the same as those in built-in evaluate.
        """

        # Adding ResetStateCallback as default
        if callbacks is None:
            callbacks = []
        callbacks.insert(0, ResetStatesCallback(self.trackers))

        return self.model.evaluate(x=x,
                                   y=y,
                                   batch_size=batch_size,
                                   verbose=self.verbose,
                                   sample_weight=sample_weight,
                                   steps=steps,
                                   callbacks=callbacks,
                                   max_queue_size=max_queue_size,
                                   workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=return_dict)

    def train_step(self, data):
        """
        The argument data represents what is yielded from tf.data.Dataset. It
        is expected to be a tuple with four elements, namely:

        features, features_size, encodeds, encodeds_size = data

        where

        - features has shape [batch_size, time, channels], and is of type
          tf.float32
        - features_size has shape [batch_size], and is of type tf.int32, and
          represents the number of time frames per example in the batch
        - encodeds has shape [batch_size, None], and is of type tf.int32, and
          represents a text encoded version of the original sentence per
          example in the batch; it contains values in the range [1,
          encoder.vocab_size)
        - encodeds_size has shape [batch_size], and is of type tf.int32, and
          represents the number of tokens in each text encoded version of the
          original sentence

        In all above batch_size and time and determined at run time, whereas
        channels is defined at compile time
        """

        metrics = {}

        features, features_size, encodeds, encodeds_size = data
        with tf.GradientTape() as tape:
            logits = self.model(features, training=True)
            logits_size = get_logits_size(features, features_size, logits)
            ctc_loss = self.loss(logits, logits_size, encodeds, encodeds_size, metrics=metrics)
            total_loss = tf.math.add_n([ctc_loss] + self.model.losses)

        # Horovod: (optional) compression algorithm.
        # compression = hvd.Compression.fp16 if self.fp16_allreduce else hvd.Compression.none
        # # Horovod: add Horovod Distributed GradientTape.
        # tape = hvd.DistributedGradientTape(tape, compression=compression)
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        grads = [
            tf.debugging.check_numerics(tensor=grad,
                                        message="nan or inf in grad",
                                        name="grad") for grad in grads
        ]
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        unused_trainable_variables = [
            tf.debugging.check_numerics(tensor=var,
                                        message="nan or inf in train_var",
                                        name="train_var")
            for var in self.model.trainable_variables
        ]


        return metrics

    def test_step(self, data):
        """
        The argument data represents what is yielded from tf.data.Dataset. It
        is expected to be a tuple with four elements, namely:

        features, features_size, encodeds, encodeds_size = data

        where

        - features has shape [batch_size, time, channels], and is of type
          tf.float32
        - features_size has shape [batch_size], and is of type tf.int32, and
          represents the number of time frames per example in the batch
        - encodeds has shape [batch_size, None], and is of type tf.int32, and
          represents a text encoded version of the original sentence per
          example in the batch; it contains values in the range [1,
          encoder.vocab_size)
        - encodeds_size has shape [batch_size], and is of type tf.int32, and
          represents the number of tokens in each text encoded version of the
          original sentence

        In all above batch_size and time and determined at run time, whereas
        channels is defined at compile time
        """

        metrics = {}

        features, features_size, encodeds, encodeds_size = data
        logits = self.model(features, training=False)
        logits_size = get_logits_size(features, features_size, logits)
        _ = self.loss(logits, logits_size, encodeds, encodeds_size, metrics=metrics)
        logits_transposed = tf.transpose(logits, [1, 0, 2])
        logits_encodeds = get_logits_encodeds(
            logits_transposed=logits_transposed,
            logits_size=logits_size,
            greedy_decoder=self.greedy_decoder,
            beam_width=self.beam_width,
        )
        # tfds.features.text.SubwordTextEncoder can only run on CPU
        with tf.device("/CPU:0"):
            sentences = tf.map_fn(self.encoder.get_decoded_from_encoded,
                                  encodeds,
                                  dtype=tf.string)
            logits_sentences = tf.map_fn(self.encoder.get_decoded_from_encoded,
                                         logits_encodeds,
                                         dtype=tf.string)

        _, _, _, _, hash_table = get_phoneme_mapping(source_enc_name='p48', dest_enc_name='p39')
        encodeds = old_to_new_indices(hash_table, encodeds)
        logits_encodeds = old_to_new_indices(hash_table, logits_encodeds)

        wer_numerator, wer_denominator = get_wer_numerator_denominator(
            sentences=sentences, logits_sentences=logits_sentences)

        ler_numerator, ler_denominator = get_ler_numerator_denominator(
            encodeds=encodeds, logits_encodeds=logits_encodeds)

        metrics.update({
            "wer": (wer_numerator, wer_denominator),
            "ler": (ler_numerator, ler_denominator),
        })

        return metrics


def get_trainer(*args, **kwargs):
    return Trainer(*args, **kwargs)
