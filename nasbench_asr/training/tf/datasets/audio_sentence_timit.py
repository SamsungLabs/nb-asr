# pylint: skip-file
import os
import glob

from nasbench_asr.quiet_tensorflow import tensorflow as tf


def get_paths_to_wav(folder):
    """
    Args:
      folder: string with location of TIMIT dataset, should end in "TRAIN" or
              "TEST"
    Returns:
      paths_to_wav: sorted list "*.RIFF.WAV" files in folder or in any of its
                    children
    """
    folder = os.path.expanduser(folder)
    assert os.path.exists(folder)
    pattern = folder + "/**/*.RIFF.WAV"
    paths_to_wav = glob.glob(pattern, recursive=True)
    # sort() ensures different calls to this function always return the same
    # list
    paths_to_wav.sort()

    return paths_to_wav


def get_audio_and_sentence_fn(encoder_class):
    @tf.function
    def get_audio_and_sentence(path_to_wav):
        """
        Args:
        path_to_wav: tf.string with path to a wav file
        Returns:
        (audio, sentence):
            - audio is tf.float32 of shape [None]
            - sentence is a tf.string of shape [], without '\n', without '.', upper
            case
        """
        # Original TIMIT audio files have Header NIST, but tf.audio.decode_wav
        # expects RIFF.  Use sox to fix this issue, namely run:
        # apt-get install -y parallel
        # find TIMIT -name '*.WAV' | parallel -P20 sox {} '{.}.RIFF.WAV'
        audio, _ = tf.audio.decode_wav(tf.io.read_file(path_to_wav),
                                       desired_channels=1,
                                       desired_samples=-1)
        audio = tf.squeeze(audio)

        if encoder_class == "phoneme":
            path_to_phn = tf.strings.join(
                [tf.strings.split(path_to_wav, sep=".RIFF.WAV")[0], ".PHN"])

            def get_last_column(sentence):
                return tf.strings.split(sentence, sep=" ")[-1]

            phonemes = tf.io.read_file(path_to_phn)
            phonemes = tf.strings.strip(phonemes)
            phonemes = tf.strings.split(phonemes, sep="\n")
            phonemes = tf.map_fn(get_last_column, phonemes)

            return audio, phonemes
        else:
            path_to_txt = tf.strings.join(
                [tf.strings.split(path_to_wav, sep=".RIFF.WAV")[0], ".TXT"])
            sentence = tf.strings.reduce_join(
                tf.strings.split(tf.io.read_file(path_to_txt), sep=" ")[2:],
                separator=" ",
            )
            # Remove '\n' from end of sentence
            sentence = tf.strings.strip(sentence)
            # Replace '.' with ''
            sentence = tf.strings.regex_replace(sentence, "\.+", "")
            # Change sentence to upper
            sentence = tf.strings.upper(sentence)

            return audio, sentence

    return get_audio_and_sentence


def get_timit_audio_sentence_ds(folder,
        ds_name,
        remove_sa=True,
        encoder_class='phoneme',
        num_parallel_calls=-1,
        deterministic=True,
        max_audio_size=0):
    """
    Returns:
      - ds: yields (audio, sentence) tuples
        where
        - audio has shape [None] and is of type tf.float32
        - sentence has shape [] and is of type tf.string
        from the TIMIT dataset indicated in the params.
    """
    paths_to_wav = get_paths_to_wav(folder=os.path.join(folder, ds_name))
    if remove_sa:
        paths_to_wav = [
            path_to_wav for path_to_wav in paths_to_wav
            if path_to_wav.split("/")[-1][:2] != "SA"
        ]
    ds = tf.data.Dataset.from_tensor_slices(paths_to_wav)
    get_audio_and_sentence = get_audio_and_sentence_fn(encoder_class=encoder_class)
    ds = ds.map(
        get_audio_and_sentence,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
    )

    if max_audio_size > 0:
        def filter_fn(audio, sentence):
            return tf.size(audio) < tf.saturate_cast(max_audio_size, tf.int32)

        ds = ds.filter(filter_fn)

    return ds
