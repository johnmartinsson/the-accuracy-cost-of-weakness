import numpy as np

def overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))

# Definition 1
def event_fraction(e, q):
    d_e = e[1] - e[0]
    o = overlap(q[0], q[1], e[0], e[1])
    return o / d_e

# Definition 2
def annotator_criterion(e, q, gamma):
    ef = event_fraction(e, q)
    return ef >= gamma

# Definition 3
def query_segment_accuracy_score(e, q, gamma):
    d_q = q[1] - q[0]
    o = overlap(q[0], q[1], e[0], e[1])
    
    if annotator_criterion(e, q, gamma):
        return o / d_q
    else:
        return (d_q-o)/d_q

def occurences_with_overlap(t, d_e, d_q, gamma, metric):
    """
    Evaluate the metric for a given t by defining the event timings relative to the query timings.
    """
    q = [0, d_q]
    e = [0 - d_e + t, 0 + t]

    return metric(e, q, gamma)


def occurences(t, d_e, T, q, gamma, metric):
    """
    Evaluate the metric for a given t by defining the event timings relative to the query timings.
    """
    offset = -d_e
    e  = [t + offset, d_e+t+offset]

    return metric(e, q, gamma)
 

# Other scoring functions that we have looked at.
def iou_score(e, q, gamma):
    d_q = q[1] - q[0]
    if annotator_criterion(e, q, gamma):
        o = overlap(q[0], q[1], e[0], e[1])
        return o / d_q
    else:
        return 0
    

def f1_score(e, q, gamma):
    d_q = q[1] - q[0]
    o = overlap(q[0], q[1], e[0], e[1])
    if annotator_criterion(e, q, gamma):
        tp = o
        fp = d_q - o
        fn = 0
        return 2*tp / (2*tp + fp + fn)
    else:
        return 0