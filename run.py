import aicrowd_helper
import train
import test

import logging
import os

import coloredlogs
coloredlogs.install(logging.DEBUG)

EVALUATION_RUNNING_ON = os.getenv('EVALUATION_RUNNING_ON', None)
EVALUATION_STAGE = os.getenv('EVALUATION_STAGE', 'all')
EXITED_SIGNAL_PATH = os.getenv('EXITED_SIGNAL_PATH', 'shared/exited')

# Hyperparameters
n_pretrain_steps = 750000
frame_skip = 3
frame_stack = 4
gpu = 0
lr = 6.25e-5
minibatch_size = 32
n_experts = 64
use_noisy_net = "before-pretraining"
n_clusters = 40

# Training Phase
if EVALUATION_STAGE in ['all', 'training']:
    aicrowd_helper.training_start()
    try:
        #train.main(frame_skip, frame_stack, gpu, lr, minibatch_size, n_experts,
        #           use_noisy_net, n_clusters)
        aicrowd_helper.training_end()
    except Exception as e:
        logging.error(e, exc_info=True)
        aicrowd_helper.training_error()


# Testing Phase
if EVALUATION_STAGE in ['all', 'testing']:
    if EVALUATION_RUNNING_ON in ['local']:
        try:
            os.remove(EXITED_SIGNAL_PATH)
        except FileNotFoundError:
            pass
    aicrowd_helper.inference_start()
    try:
        test.main(frame_skip, frame_stack, gpu, n_clusters, use_noisy_net)
        aicrowd_helper.inference_end()
    except Exception as e:
        logging.error(e, exc_info=True)
        aicrowd_helper.inference_error()
    if EVALUATION_RUNNING_ON in ['local']:
        from pathlib import Path
        Path(EXITED_SIGNAL_PATH).touch()

# Launch instance manager
if EVALUATION_STAGE in ['manager']:
    from minerl.env.malmo import launch_instance_manager
    launch_instance_manager()
