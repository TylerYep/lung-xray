from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict

def parse(path):
    """Takes an events file and outputs a dictionary mapping a metric to a list of values"""
    d = defaultdict(list)
    for e in summary_iterator(path):
        for v in e.summary.value:
            d[v.tag].append(v.simple_value)
    return d
