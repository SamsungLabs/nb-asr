def disable_warnings():
    import os
    import logging
    import warnings
    warnings.filterwarnings('ignore',category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    try:
        from tensorflow.python.util import module_wrapper as deprecation
    except ImportError:
        try:
            from tensorflow.python.util import deprecation_wrapper as deprecation
        except ImportError:
            from tensorflow.python.util import deprecation
    try:
        deprecation._PRINT_DEPRECATION_WARNINGS = False
    except:
        pass

    import tensorflow as tf
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except:
        pass

    try:
        tf.get_logger().setLevel('ERROR')
    except:
        pass

disable_warnings()
import tensorflow
