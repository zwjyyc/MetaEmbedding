import argparse

from learner import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embs",
                        nargs='*',
                        help="pathes to embs"
                        )
    parser.add_argument("--dict",
                        type=str,
                        help="path to word dict"
                        )
    parser.add_argument("--out",
                        type=str,
                        default="",
                        help="path to data"
                        )
    parser.add_argument("--dim",
                        type=int,
                        default=128,
                        help="number of hidden units"
                        )
    parser.add_argument("--batch",
                        type=int,
                        default=256,
                        help="mini-batch size"
                        )
    parser.add_argument("--dropout_rate",
                        type=float,
                        default=0,
                        help="dropout rate"
                        )
    parser.add_argument("--eval_period",
                        type=int,
                        default=1000,
                        help="evaluate on dev every period"
                        )
    parser.add_argument("--learning",
                        type=str,
                        default="sgd",
                        help="learning method (sgd, adagrad, adam, ...)"
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default="0.001",
                        help="learning rate"
                        )
    parser.add_argument("--l2_reg",
                        type=float,
                        default=0.00001
                        )
    parser.add_argument("--max_epochs",
                        type=int,
                        default=100,
                        help="maximum # of epochs"
                        )
    parser.add_argument("--save",
                        type=str,
                        default="models/DrModel",
                        help="save model to this file"
                        )
    args = parser.parse_args()
    print args
    learner = Learner(args)

    learner.run()
