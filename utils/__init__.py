
import sys
import gzip

import numpy as np


def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        cnt = 0
        for line in fin:
            line = line.strip()
            cnt += 1
            if cnt == 1:
                continue
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals

