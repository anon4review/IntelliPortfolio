from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from multiprocessing import Process
from pgportfolio.learn.tradertrainer import TraderTrainer
from pgportfolio.tools.configprocess import load_config


def train_one(config, index, device):
    """
    train an agent
    :param config: the json configuration file
    :param index: identifier of this train, which is also the sub directory in the train_package,
    if it is 0. nothing would be saved into the summary file.
    :param device: 0 or 1 to show which gpu to use, if 0, means use cpu instead of gpu
    :return : the Result namedtuple
    """
    print("training at %s started" % index)
    return TraderTrainer(config,  device=device).train_net()

def train_all(processes=1, device="cpu"):
    """
    train all the agents in the train_package folders

    :param processes: the number of the processes. If equal to 1, the logging level is debug
                      at file and info at console. If gRollingTrainerreater than 1, the logging level is
                      info at file and warming at console.
    """
    train_dir = "train_package"
    if not os.path.exists("./" + train_dir): #if the directory does not exist, creates one
        os.makedirs("./" + train_dir)
    all_subdir = os.listdir("./" + train_dir)
    all_subdir.sort()
    pool = []
    for dir in all_subdir:
        if not str.isdigit(dir):
            return
        p = Process(target=train_one, args=(
            load_config(dir), dir,device))
        p.start()
        pool.append(p)

        # suspend if the processes are too many
        wait = True
        while wait:
            time.sleep(5)
            for p in pool:
                alive = p.is_alive()
                if not alive:
                    pool.remove(p)
            if len(pool)<processes:
                wait = False
    print("All the Tasks are Over")
