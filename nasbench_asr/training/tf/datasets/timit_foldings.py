
import pathlib

from nasbench_asr.quiet_tensorflow import tensorflow as tf


def old_to_new_indices(table, old_indices):
    """ 
        This function uses lookup table to convert given indexices to corresponding ones
        
        The input to this function has shape [batch_size, max_num_indices] and is given by
        old_indices = = [old_indices[0], ..., old_indices[batch_size - 1]]
        and the output to this function has the same shape as is given by
        tf.map_fn(fn=fn, elems=old_indices) = [fn(old_indices[0]), ..., fn(old_indices[batch_size - 1])]
        As an example, let's say that table was constructed from
        - keys: [1, 2, 3]
        - vals: [3, 5, 2]
        - missing elements replaced with value 0
        and that the input is
        old_indices = [[1, 2, 44, 2, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
        Then for the first row of the input, we'll have
        x = old_indices[0] = [1, 2, 44, 2, 1, 0, 0]
        y = [3, 5, 0, 5, 3, 0, 0]
        tf.boolean_mask(y, y > 0) = [3, 5, 5, 3]
        tf.zeros(tf.reduce_sum(tf.cast(y <= 0, dtype=tf.int32)) = [0, 0, 0]
        z = [3, 5, 5, 3, 0, 0, 0]
        whereas for the second row of the input we'll have
        x = old_indices[1] = [1, 1, 1, 1, 1, 1, 1]
        y = [3, 3, 3, 3, 3, 3, 3]
        tf.boolean_mask(y, y > 0) = [3, 3, 3, 3, 3, 3, 3]
        tf.zeros(tf.reduce_sum(tf.cast(y <= 0, dtype=tf.int32)) = []
        z = [3, 3, 3, 3, 3, 3, 3]
        Therefore the output will be
        [[3, 5, 5, 3, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3]]
    """
    def fn(x):
        y = table.lookup(x)
        z = tf.concat(
            [
                tf.boolean_mask(y, y > 0),
                tf.zeros(tf.reduce_sum(tf.cast(y <= 0, dtype=tf.int32)),
                         dtype=y.dtype),
            ],
            axis=0,
        )

        return z 

    return tf.map_fn(fn=fn, elems=old_indices)

def get_phoneme_mapping(source_enc_name='p61', dest_enc_name='p48'):
    #creates a mapping [defined in timit_foldings.txt] for a given source to destination folding
    #also returns a lookup table for mapping source to dest indices
    file_path = pathlib.Path(__file__).parents[2].joinpath('timit_folding.txt')
    assert file_path.exists(), f'Timit mapping file not found: {file_path}'

    with file_path.open('r') as f:
        mapping = f.readlines()
    mapping = [m.strip().split('\t') for m in mapping]

    #remove phonemes with no mapping
    no_map_phonemes = [m[0] for m in mapping if len(m)<2]
    mapping = [m for m in mapping if m[0] not in no_map_phonemes]

    foldings = ['p61', 'p48', 'p39'] #don't change the order. It is same as the order of mapping.
    assert source_enc_name in foldings and dest_enc_name in foldings, 'Encoding name is incorrect'
    ph61 = sorted(list(set([m[0] for m in mapping] + no_map_phonemes)))
    ph48 = sorted(list(set([m[1] for m in mapping])))
    ph39 = sorted(list(set([m[2] for m in mapping])))
    phonemes = [ph61, ph48, ph39] 

    source_idx = foldings.index(source_enc_name)
    dest_idx = foldings.index(dest_enc_name)
    source_phonemes = phonemes[source_idx]
    dest_phonemes = phonemes[dest_idx]

    source_encodes = []
    dest_encodes = []
    for idx, ph in enumerate(source_phonemes):
        source_encodes.append(idx + 1)
        ph_idx_in_map = [i for i, _map in enumerate(mapping) if _map[source_idx] == ph]

        if len(ph_idx_in_map) == 0: #phoneme not resent in mapping [special case for q]
            dest_encodes.append(0)
        else:
            dest_ph = mapping[ph_idx_in_map[0]][dest_idx]
            dest_encodes.append(dest_phonemes.index(dest_ph) + 1)
    
    # create the hash table (must have 0 to 0 mapping)
    source_encodes = tf.constant(source_encodes, dtype=tf.int32)
    dest_encodes = tf.constant(dest_encodes, dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(source_encodes, dest_encodes), 0)

    return source_phonemes, source_encodes, dest_phonemes, dest_encodes, table
