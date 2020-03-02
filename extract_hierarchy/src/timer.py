#-*-coding:utf-8-*-
from time import time
from yaspin import yaspin


class Timer:
    def __init__(self):
        self.time = 0
        self.yaspin = yaspin()

    def __enter__(self):
        self.time = time()
        self.yaspin.__enter__()

    def __exit__(self, exc_t, exc_v, trace):
        self.yaspin.__exit__(exc_t, exc_v, trace)
        print("Elapsed %s seconds" % (time() - self.time))