import numpy as np

import logger
from libs.misc.data_handling.normalization import Normalization
from libs.misc.data_handling.path_collection import PathCollection


def add_path_data_to_collection_and_update_normalization(
        paths, path_collection,
        train_collection, val_collection,
        normalization=None,
        split_ratio=0.666667,
        train_discard_ratio=0.0,
        obs_dim=None,
        normalization_scope=None):
    """
    Add new data from paths to collections. Update normalization stats.
    :param path_collection: PathCollection object
    :param train_collection: a "data_collection" object
    :param val_collection: a "data_collection" object
    :param normalization: a "Normalization" object
    :param split_ratio: real number in [0, 1]. The split ratio between training and validation set
    :param train_discard_ratio: real number in [0, 1). The ratio to discard training data
    :param obs_dim: actual observation dimension that will be normalized
    :return: an updated normalization
    """
    # data
    train_data, val_data = PathCollection.to_data_collections(paths, split_ratio=split_ratio)
    train_collection.add_data(train_data, discard_ratio=train_discard_ratio)
    val_collection.add_data(val_data)

    # normalization
    if not normalization:
        logger.log("Creating normalization for training data.")
        normalization = Normalization(train_data, obs_dim=obs_dim, scope=normalization_scope)
        logger.log("Done creating normalization for training data.")
        return normalization
    else:
        logger.log("Updating normalization.")
        normalization.update(train_collection.get_data())
        logger.log("Done updating normalization.")
        return normalization


def replace_path_data_to_collection_and_update_normalization(
        paths, train_collection, val_collection, normalization=None,
        split_ratio=0.666667, obs_dim=None, normalization_scope=None):
    # data
    train_data, val_data = PathCollection.to_data_collections(paths, split_ratio=split_ratio)
    train_collection.replace_data(train_data)
    val_collection.replace_data(val_data)

    # normalization
    if not normalization:
        logger.log("Creating normalization for training data.")
        normalization = Normalization(train_data, obs_dim=obs_dim, scope=normalization_scope)
        logger.log("Done creating normalization for training data.")
        return normalization
    else:
        logger.log("Updating normalization.")
        normalization.update(train_collection.get_data())
        logger.log("Done updating normalization.")
        return normalization


def get_normalized_iteration(itr, itr_mean, itr_std):
    return (itr - itr_mean) / itr_std


def get_iteration_stats(min_itr, max_itr):
    iterations = list(range(min_itr, max_itr))
    return np.mean(iterations), max(np.asscalar(np.std(iterations)), 0.5)
