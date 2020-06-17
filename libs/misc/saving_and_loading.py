import os
import pickle
import time
import pickle

import logger


def save_cur_iter_dynamics_model(params, saver, sess, itr):
    if params.get("save_variables"):
        save_path = saver.save(sess, os.path.join(params['exp_dir'], "model-iter{}.ckpt".format(itr)))
        logger.log("Model saved in path {}".format(save_path))


def confirm_restoring_dynamics_model(params):
    return params.get("restore_dynamics_variables", False) and params.get("restore_path", False)


def restore_model(params, saver, sess):
    restore_path = os.path.join(params["restore_path"], "model-iter{}.ckpt".format(params['restart_iter']))
    saver.restore(sess, restore_path)
    logger.log("Model restored from {}".format(restore_path))


def save_cur_iter_behavior_policy(params, saver, sess, itr):
    if params.get("save_variables"):
        save_path = saver.save(sess, os.path.join(params['exp_dir'], "behavior_policy-iter{}.ckpt".format(itr)))
        logger.log("Model saved in path {}".format(save_path))


def confirm_restoring_behavior_policy(params):
    return params.get("restore_bc_variables", False) and params.get("restore_path", False)


def restore_behavior_policy(params, saver, sess):
    restore_path = os.path.join(params['restore_path'], "behavior_policy-iter{}.ckpt".format(params['restart_iter']))
    saver.restore(sess, restore_path)
    logger.log("Behavior policy restored from {}".format(restore_path))


def save_cur_iter_policy(params, saver, sess, itr):
    if params.get("save_variables"):
        save_path = saver.save(sess, os.path.join(params['exp_dir'], "policy-iter{}.ckpt".format(itr)))
        logger.log("Model saved in path {}".format(save_path))


def confirm_restoring_policy(params):
    return params.get("restore_policy_variables", False) and params.get("restore_path", False)


def restore_policy(params, saver, sess):
    restore_path = os.path.join(params['restore_path'], "policy-iter{}.ckpt".format(params['restart_iter']))
    saver.restore(sess, restore_path)
    logger.log("Policy restored from {}".format(restore_path))


def restore_policy_for_video(restore_path, saver, sess):
    saver.restore(sess, restore_path)
    logger.log("Policy restored from {}".format(restore_path))


def save_cur_iter_offline_data(params, train, val, bc_train, itr):
    if params.get("save_variables"):
        with open(os.path.join(params['exp_dir'], 'train_collection_{}.pickle'.format(itr)), 'wb') as f:
            pickle.dump(train, f)
        with open(os.path.join(params['exp_dir'], 'val_collection_{}.pickle'.format(itr)), 'wb') as f:
            pickle.dump(val, f)
        with open(os.path.join(params['exp_dir'], 'behavior_policy_train_collection_{}.pickle'.format(itr)), 'wb') as f:
            pickle.dump(bc_train, f)


def restore_offline_data(params):
    with open(os.path.join(params['restore_path'], 'train_collection_{}.pickle'.format(params['restart_iter'])), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(params['restore_path'], 'val_collection_{}.pickle'.format(params['restart_iter'])), 'rb') as f:
        test = pickle.load(f)
    with open(os.path.join(params['restore_path'], 'behavior_policy_train_collection_{}.pickle'.format(params['restart_iter'])), 'rb') as f:
        bc_train = pickle.load(f)
    return train, test, bc_train


def confirm_restoring_offline_data(params):
    return params.get("restore_offline_data", False) and params.get("restore_path", False)


def confirm_restoring_value(params):
    return params.get("restore_value", False) and params.get("restore_path", False)
