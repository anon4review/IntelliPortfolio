from __future__ import absolute_import
import logging
import os
from argparse import ArgumentParser
from pgportfolio.tools.configprocess import load_config

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="start mode, train, generate, download_data"
                             " backtest",
                        metavar="MODE", default="train")
    parser.add_argument("--processes", dest="processes",
                        help="number of processes you want to start to train the network",
                        default="1")
    parser.add_argument("--repeat", dest="repeat",
                        help="repeat times of generating training subfolder",
                        default="1")
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + "train_package"):
        os.makedirs("./" + "train_package")

    if options.mode == "train":
        import pgportfolio.autotrain.training
        pgportfolio.autotrain.training.train_all(int(options.processes))
    elif options.mode == "generate":
        import pgportfolio.autotrain.generate as generate
        logging.basicConfig(level=logging.INFO)
        generate.add_packages(load_config(), int(options.repeat))

if __name__ == "__main__":
    main()
