# coding=utf-8

"""
Collection of function to extract features from from audio signal in TensorFlow
Author: SpeechX team, Cambrigde
"""

import numpy as np
from nasbench_asr.quiet_tensorflow import tensorflow as tf



def normalize_audio_full_scale(audio_pcm):
    """
    Normalizes audio data to the full scale
    """
    max_abs_val = tf.math.reduce_max(tf.math.abs(audio_pcm))

    # Estimating the scale factor
    gain = tf.constant(1.0) / (max_abs_val + tf.constant(1e-5))

    return audio_pcm * gain


def normalize_spectrogram(magnitude_spectrogram):
    """
    Performs mean and variance normalization of a spectrogram
    """

    magnitude_spectrogram -= tf.math.reduce_mean(magnitude_spectrogram)
    magnitude_spectrogram /= tf.math.reduce_std(magnitude_spectrogram)

    return magnitude_spectrogram


def convert_to_dB(magnitude_spectrogram, normalize=False, ref_level_db=20.0, min_level_db=-100.0):
    """
    Converts spectrograms to dB

    Args:
        magnitude_spectrogram : A tensor containing magnitude spectrogram
        normalize :             A bool indicating if spectrogram normalization is needed
        ref_level_db:           Ref db level required [default 20]. Pass None if this not to be used.
        min_level_db:           Minimum db level required [default -100]. Pass None if this not to be used.
    """

    # Removing small values before taking log
    magnitude_spectrogram = tf.clip_by_value(magnitude_spectrogram,
            clip_value_min=1e-30,
            clip_value_max=tf.math.reduce_max(magnitude_spectrogram))

    magnitude_spectrogram = 10.0 * tf.math.log(magnitude_spectrogram) / tf.math.log(10.0)

    if normalize:
        magnitude_spectrogram = normalize_spectrogram(magnitude_spectrogram)

    # this is from amp-to-db function in librosa
    # Source: https://librosa.org/librosa/master/_modules/librosa/core/spectrum.html#amplitude_to_db
    # Source: https://github.com/mindslab-ai/voicefilter/blob/master/utils/audio.py
    if ref_level_db is not None:
        magnitude_spectrogram -= ref_level_db
    if min_level_db is not None:
        magnitude_spectrogram /= -min_level_db
        magnitude_spectrogram = tf.clip_by_value(magnitude_spectrogram, clip_value_min=-1.0, clip_value_max=0.0) + 1.0

    return magnitude_spectrogram


def get_stft(audio_pcm,
             normalize=False,
             fft_length=512,
             window_len=None,
             step_len=None,
             center=True,
             verbose=0):

    """
    Performs short time fourier transformation of a time domain audio signal

    Parameters
    ----------
    audio_pcm :     A 1D tensor (float32) holding the input audio
    fft_length :    (int in samples) length of the windowed signal after padding,
                    which will be used to extract FFT
    window_len :    (int > 0 and <= fft_length) length of each audio frame in samples [default: fft_length]
    step_len :      (int > 0) length of hop / stride in samples [default: window_length // 4]
    center :        (Bool) Type of padding to be used to match librosa
    verbose :       Verbosity level, 0 = no ouput, > 0 debug prints

    This function returns a complex-valued matrix stfts
    """

    # Checking the input type and perform casting if necessary
    if audio_pcm.dtype != 'float32':
        audio_pcm = tf.cast(audio_pcm, tf.float32)

    # Performing audio normalization
    if normalize:
        audio_pcm = normalize_audio_full_scale(audio_pcm)

    if window_len is None:
        window_len = fft_length

    if step_len is None:
        step_len = int(window_len // 4)

    # Perform padding of the original signal
    if center:
        pad_amount = int(window_len // 2) # As used by Librosa

        if verbose > 0:
            print(f'[INFO] (audio_feature.get_stft)] pad_amount = {pad_amount}')

        audio_pcm = tf.pad(audio_pcm, [[pad_amount, pad_amount]], 'REFLECT')

    # Extracting frames from sudio signal
    frames = tf.signal.frame(audio_pcm, window_len, step_len, pad_end=False)

    if verbose > 0:
        print(f'[INFO] (audio_feature.get_stft)] frames.shape = {frames.shape}')

    # Generating hanning window
    fft_window = tf.signal.hann_window(window_len, periodic=True)

    # Computing the spectrogram, the output is an array of complex number
    stfts = tf.signal.rfft(frames * fft_window, fft_length=[fft_length])

    return stfts


def get_magnitude_spectrogram(audio_pcm,
                              sample_rate,
                              window_len_in_sec=0.025,
                              step_len_in_sec=0.010,
                              exponent=2.0,
                              nfft=None,
                              normalize_full_scale=False,
                              compute_phase=False,
                              verbose=0):

    """
    Computes the magnitude spectrogram of an audio signal

    Parameters:
    -----------
    audio_pcm:              A 1D tensor (float32) holding the input audio
    sample_rate:            Samling frequency of the recorded audio
    window_len_in_sec:      float, in seconds
    step_len_in_sec:        float, in seconds
    exponent:               Int, 1 for energy and 2 for power [default 2]
    normalize_full_scale:   If full scale power normalization to be
                            performed, default is False
    compute_phase           Compute and return phase for all frames [default False]
    verbose:                Verbosity level, 0 = no ouput, > 0 debug prints
    """

    # Full-scale normalization of audio
    if normalize_full_scale:
        audio_pcm = normalize_audio_full_scale(audio_pcm)

    # Estimating parameters for STFT
    frame_length_in_sample = int(window_len_in_sec * sample_rate)
    frame_step_in_sample = int(step_len_in_sec * sample_rate)

    if nfft is None:
        nfft = frame_length_in_sample

    stfts = tf.signal.stft(
                signals=audio_pcm,
                frame_length=frame_length_in_sample,
                frame_step=frame_step_in_sample,
                fft_length=nfft,
                window_fn=tf.signal.hann_window,
                pad_end=False)

    magnitude_spectrograms = tf.abs(stfts)

    if exponent != 1.0:
        magnitude_spectrograms = tf.math.pow(magnitude_spectrograms, exponent)

    phases = None
    if compute_phase:
        phases = tf.math.angle(stfts)

    return magnitude_spectrograms, phases


def get_magnitude_spectrogram_dB(audio_pcm,
                                 sample_rate,
                                 normalize_spec=False,
                                 ref_level_db=20.0,
                                 min_level_db=-100.0,
                                 **kwargs):
    """
    Computes the magnitude spectrogram in dB

    Parameters:
    -----------
    audio_pcm:              A 1D tensor (float32) holding the input audio
    sample_rate:            Samling frequency of the recorded audio
    window_len_in_sec:      float, in seconds
    step_len_in_sec:        float, in seconds
    exponent:               Int, 1 for energy and 2 for power [default 2]
    normalize_full_scale:   If full scale power normalization to be
                            performed, default is False
    normalize_spec          Is magnitude spectogram to be normalized [default False]
    ref_level_db:           Ref db level required [default 20]
    min_level_db:           Minimum db level required [default -100]
    compute_phase           Compute and return phase for all frames [default False]
    verbose:                Verbosity level, 0 = no ouput, > 0 debug prints
    """


    magnitude_spectrogram, phase = get_magnitude_spectrogram(audio_pcm, sample_rate, **kwargs)
    magnitude_spectrogram_dB = convert_to_dB(magnitude_spectrogram, normalize=normalize_spec, ref_level_db=ref_level_db, min_level_db=min_level_db)

    return magnitude_spectrogram_dB, phase


def wav2spec(audio_pcm,
            sample_rate,
            ref_level_db=20.0,
            min_level_db=-100.0,
            **kwargs):
    """
    Computes the magnitude spectrogram in dB and phase

    Parameters:
    -----------
    audio_pcm:              A 1D tensor (float32) holding the input audio
    sample_rate:            Samling frequency of the recorded audio
    ref_level_db:           Ref db level required [default 20]
    min_level_db:           Minimum db level required [default -100]
    window_len_in_sec:      float, in seconds
    step_len_in_sec:        float, in seconds
    exponent:               Int, 1 for energy and 2 for power [default 2]
    normalize_full_scale:   If full scale power normalization to be
                            performed, default is False
    verbose:                Verbosity level, 0 = no ouput, > 0 debug prints
    """

    magnitude_spectrogram, phase = get_magnitude_spectrogram(audio_pcm, sample_rate, **dict(kwargs, compute_phase=True))
    magnitude_spectrogram_dB = convert_to_dB(magnitude_spectrogram, normalize=False, ref_level_db=ref_level_db, min_level_db=min_level_db)
    return magnitude_spectrogram_dB, phase


def spec2wav(magnitude_spectrogram,
            phase,
            sample_rate,
            nfft=None,
            ref_level_db=20,
            min_level_db=-100,
            window_len_in_sec=0.025,
            step_len_in_sec=0.010,
            exponent=2.0):
    """
    Computes the audio pcm from magnitude spectrogram and phase

    Parameters:
    -----------
    magnitude_spectrogram:  Magnitude spectogram of audio pcm
    phase:                  Phase obtained from stfts of audio pcm
    sample_rate:            Samling frequency of the recorded audio
    ref_level_db:           Ref db level required [defaul 20]
    min_level_db:           Minimum db level required [default -100]
    window_len_in_sec:      float, in seconds
    step_len_in_sec:        float, in seconds
    exponent:               Int, 1 for energy and 2 for power [default 2]
    """

    magnitude_spectrogram = tf.clip_by_value(magnitude_spectrogram, clip_value_min=0.0, clip_value_max=1.0)
    magnitude_spectrogram = (magnitude_spectrogram - 1.0) * - min_level_db
    magnitude_spectrogram += ref_level_db
    magnitude_spectrogram = tf.math.pow(tf.constant(10.0), magnitude_spectrogram / (exponent*10))
    magnitude_spectrogram = tf.cast(magnitude_spectrogram, dtype=tf.complex64)

    phase = tf.complex(tf.zeros(tf.shape(phase)), phase)
    phase = tf.math.exp(phase)
    stfts = magnitude_spectrogram * phase

    # Estimating parameters for STFT
    frame_length_in_sample = int(window_len_in_sec * sample_rate)
    frame_step_in_sample = int(step_len_in_sec * sample_rate)
    if nfft is None:
        nfft = frame_length_in_sample

    W = tf.signal.inverse_stft(
                            stfts=stfts,
                            frame_length=frame_length_in_sample,
                            frame_step=frame_step_in_sample,
                            fft_length=nfft,
                            window_fn=tf.signal.inverse_stft_window_fn(
                                frame_step=frame_step_in_sample,
                                forward_window_fn=tf.signal.hann_window
                                )
                            )
    return W

def get_mel_filterbank(audio_pcm,
                       sample_rate,
                       window_len_in_sec=0.025,
                       step_len_in_sec=0.010,
                       nfft=None,
                       normalize_full_scale=False,
                       num_feature_filters=40,
                       lower_edge_hertz=0.0,
                       upper_edge_hertz=8000.0,
                       exponent=2.0,
                       mel_weight_mat=None,
                       verbose=0):
    """
    Computes Mel-filterbank features from an audio signal using TF operations.

    Parameters
    ----------
    audio_pcm:              A 1D tensor (float32) holding the input audio
    sample_rate:            Samling frequency of the recorded audio
    window_len_in_sec:      float, in seconds
    step_len_in_sec:        float, in seconds
    num_feature_filters:    int, e.g., 40
    lower_edge_hertz:       Lower frequency to consider in the mel scale
    upper_edge_hertz:       Upper frequency to consider in the mel scale
    exponent:               Int, 1 for energy and 2 for power [default 2]
    mel_weight_mat:         Accepts a mel_weight_matrix [defult None and generates using the HTK algo]
    verbose:                Verbosity level, 0 = no ouput, > 0 debug prints

    Returns:
        Tensor (audio_len // int(step_len * sample_rate), num_feature_filters), float32
    """

    magnitude_spectrograms, _ = \
            get_magnitude_spectrogram(
                    audio_pcm=audio_pcm,
                    sample_rate=sample_rate,
                    window_len_in_sec=window_len_in_sec,
                    step_len_in_sec=step_len_in_sec,
                    nfft=nfft,
                    compute_phase=False,
                    exponent=exponent,
                    normalize_full_scale=normalize_full_scale,
                    verbose=verbose)

    if verbose:
        print('[INFO] (audio_feature.get_mel_filterbank) magnitude_spectrograms.shape', magnitude_spectrograms.shape)

    num_spectrogram_bins = tf.shape(magnitude_spectrograms)[-1]

    if mel_weight_mat is None:
        if verbose:
            print('[INFO] (audio_feature.get_mel_filterbank) mel_weight_mat not provided and generating using TF2.')
        mel_weight_mat = tf.signal.linear_to_mel_weight_matrix(num_feature_filters,
                                                               num_spectrogram_bins,
                                                               sample_rate,
                                                               lower_edge_hertz,
                                                               upper_edge_hertz)

    if verbose:
        print('[INFO] (audio_feature.get_mel_filterbank) linear_to_mel_weight_matrix.shape', mel_weight_mat.shape)


    mel_spectrograms = tf.tensordot(magnitude_spectrograms, mel_weight_mat, 1)
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(mel_weight_mat.shape[-1:])
        )

    if verbose:
        print('[INFO] (audio_feature.get_mel_filterbank) mel_spectrograms.shape', mel_spectrograms.shape)

    return mel_spectrograms


def get_log_mel_filterbank(audio_pcm,
                           sample_rate,
                           **kwargs):
    """
    Computes Log-Mel-filterbank features from an audio signal using TF operations.

    Parameters
    ----------
    audio_pcm:              A 1D tensor (float32) holding the input audio
    sample_rate:            Samling frequency of the recorded audio
    window_len_in_sec:      float, in seconds
    step_len_in_sec:        float, in seconds
    num_feature_filters:    Int, e.g., 40
    lower_edge_hertz:       Lower frequency to consider in the mel scale
    upper_edge_hertz:       Upper frequency to consider in the mel scale
    exponent:               Int, 1 for energy and 2 for exponent [default 2]
    mel_weight_mat:         Accepts a mel_weight_matrix [defult None and generates using the HTK algo]
    verbose:                Verbosity level, 0 = no ouput, > 0 debug prints

    Returns:
        numpy.ndarray, (audio_len // int(step_len * sample_rate), num_feature_filters), float32
    """
    return tf.math.log(get_mel_filterbank(audio_pcm, sample_rate, **kwargs) + 1e-10)

def get_mfcc(audio_pcm,
             sample_rate,
             **kwargs):
    """
    Extracts mfcc feature from audio_pcm measurements

    Parameters
    ----------
    audio_pcm:              A 1D tensor (float32) holding the input audio
    sample_rate:            Samling frequency of the recorded audio
    window_len_in_sec:      float, in seconds
    step_len_in_sec:        float, in seconds
    num_feature_filters:    int, e.g., 40
    lower_edge_hertz:       Lower frequency to consider in the mel scale
    upper_edge_hertz:       Upper frequency to consider in the mel scale
    exponent:               Int, 1 for energy and 2 for power [default 2]
    mel_weight_mat:         Accepts a mel_weight_matrix [defult None and generates using the HTK algo]
    verbose:                Verbosity level, 0 = no ouput, > 0 debug prints

    """

    log_mel_spectrograms = get_log_mel_filterbank(audio_pcm, sample_rate, **kwargs)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)

    return mfccs


def get_power_mel_filterbank(audio_pcm,
                             sample_rate,
                             power_coeff=1.0 / 15.0,
                             **kwargs):
    """
    Computes power Mel-filterbank features (PNCC) from an audio signal.

    References:
    https://github.sec.samsung.net/STAR/speech/blob/master/speech/trainer/returnn_based_end_to_end_trainer/ver0p2/GeneratingDataset.py

    Parameters
    ----------
    audio_pcm:              A 1D tensor (float32) holding the input audio
    sample_rate:            Samling frequency of the recorded audio
    window_len_in_sec:      float, in seconds
    step_len_in_sec:        float, in seconds
    num_feature_filters:    Int, e.g., 40
    lower_edge_hertz:       Lower frequency to consider in the mel scale
    upper_edge_hertz:       Upper frequency to consider in the mel scale
    exponent:               Int, 1 for energy and 2 for power [default 2]
    mel_weight_mat:         Accepts a mel_weight_matrix [defult None and generates using the HTK algo]
    verbose:                Verbosity level, 0 = no ouput, > 0 debug prints

    Returns:
       Tensor, (audio_len // int(step_len * sample_rate), num_feature_filters), float32
    """
    assert power_coeff > 0.0 and power_coeff < 1.0, 'Invalid power_coeff!!!'

    mel_filterbank = get_mel_filterbank(audio_pcm, sample_rate, **kwargs)

    feature_vector = mel_filterbank ** power_coeff

    return feature_vector

def get_feature(audio, sample_rate, feature_type='pmel', **kwargs):
    """
    A wrapper function for audio features
    """
    if feature_type == 'spec':
        return get_magnitude_spectrogram(audio, sample_rate, **kwargs)
    elif feature_type == 'spec_dB':
        return get_magnitude_spectrogram_dB(audio, sample_rate, **kwargs)
    elif feature_type == 'pmel':
        return get_power_mel_filterbank(audio, sample_rate, **kwargs)
    elif feature_type == 'lmel':
        return get_log_mel_filterbank(audio, sample_rate, **kwargs)
    elif feature_type == 'mel':
        return get_mel_filterbank(audio, sample_rate, **kwargs)
    elif feature_type == 'mfcc':
        return get_mfcc(audio, sample_rate, **kwargs)
    else:
        raise NotImplementedError(f'Unsupported audio feature type {feature_type}')
