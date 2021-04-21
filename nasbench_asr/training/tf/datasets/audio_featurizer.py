# pylint: skip-file
import os
import sys

from .audio_feature import get_feature


class AudioFeaturizer:
    def __init__(self, sample_rate, feature_type, normalize_full_scale,
                 window_len_in_sec, step_len_in_sec, num_feature_filters,
                 mel_weight_mat, verbose):
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.normalize_full_scale = normalize_full_scale
        self.window_len_in_sec = window_len_in_sec
        self.step_len_in_sec = step_len_in_sec
        self.num_feature_filters = num_feature_filters
        self.mel_weight_mat = mel_weight_mat
        self.verbose = verbose

    def namespace(self):
        """
        This function returns a list of the hyper-parameters related to
        transformation of audio into features, which impacts the creation of
        the caches of the datasets that we support (grep also for "def
        get_path_ds_cache" to find where this is used toward that end).
        """
        output = ""

        params = [
            "sample_rate",
            "feature_type",
            "normalize_full_scale",
            "window_len_in_sec",
            "step_len_in_sec",
            "num_feature_filters",
            "mel_weight_mat",
        ]

        for param in params:
            output += param + "_" + \
                str(getattr(self, param)).replace(
                    "/", "_").replace("*", "_") + "/"

        return output

    def __call__(self, audio):
        if self.feature_type in ["spec", "spec_dB"]:
            audio_feature, _ = get_feature(
                audio,
                sample_rate=self.sample_rate,
                feature_type=self.feature_type,
                normalize_full_scale=self.normalize_full_scale,
                window_len_in_sec=self.window_len_in_sec,
                step_len_in_sec=self.step_len_in_sec,
                verbose=self.verbose)
        else:
            audio_feature = get_feature(
                audio,
                sample_rate=self.sample_rate,
                feature_type=self.feature_type,
                normalize_full_scale=self.normalize_full_scale,
                window_len_in_sec=self.window_len_in_sec,
                step_len_in_sec=self.step_len_in_sec,
                num_feature_filters=self.num_feature_filters,
                mel_weight_mat=self.mel_weight_mat,
                verbose=self.verbose)

        return audio_feature
