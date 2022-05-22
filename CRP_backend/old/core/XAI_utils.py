from CRP_backend.core.layer_specifics import sum_relevance

import numpy as np


def reduce_ch_accuracy(rel_layer, accuracy=0.90):
    """
       returns the most relevant channels so that the sum of their relevances is bigger or equal to accuracy * summed relevance

       relevance_layer without batch dimension
    """

    if 0 > accuracy or 1 < accuracy:
        raise ValueError("<accuracy> must be between 0 and 1.")

    abs_rel_layer = abs(rel_layer)
    rel_summed = np.sum(abs_rel_layer, axis=-1)
    max_rel = rel_summed * accuracy

    indices_ch = np.flip(np.argsort(abs_rel_layer, axis=-1))

    rel = 0
    for i, ch in enumerate(indices_ch):
        rel += abs_rel_layer[ch]
        if rel >= max_rel:
            return indices_ch[:i + 1], rel_layer[indices_ch][:i + 1]

    return indices_ch, rel_layer

