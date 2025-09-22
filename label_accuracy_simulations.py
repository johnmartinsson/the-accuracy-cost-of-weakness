import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glob
import sys
from functools import partial
from intervaltree import Interval, IntervalTree

import argparse


from scipy.stats import gamma as gamma_pdf

import definitions as defs
import theorems as thms

def load_event_lengths(class_name):
    event_lengths = np.load(f"{class_name}_event_lengths.npy")
    return event_lengths

# merge overlapping events in O(n log n) time
def merge_intervals(intervals):
    # Sort the intervals by their start time
    intervals.sort(key=lambda x: x[0])

    # The merged list of intervals
    merged = []

    for interval in intervals:
        # If the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Otherwise, there is overlap, so we merge the current and previous intervals.
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

    return merged


class Distribution:
    def __init__(self, event_lengths, mi_event_length, ma_event_length):
        event_lengths = np.array(event_lengths)
        event_lengths = event_lengths[event_lengths >= mi_event_length]
        event_lengths = event_lengths[event_lengths <= ma_event_length]
        self.event_lengths = event_lengths
        assert len(self.event_lengths) > 0

    def sample(self):
        return np.random.choice(self.event_lengths)
    
    def max(self):
        return np.max(self.event_lengths)

    def min(self):
        return np.min(self.event_lengths)


# get the standard color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

recording_length = 100
mi_event_length = 0.01
ma_event_length = recording_length
    
# different distributions
uniform_1_1 = Distribution(np.random.uniform(1, 1, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)
uniform_02_02 = Distribution(np.random.uniform(0.2, 0.2, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)

dog_event_lengths = Distribution(load_event_lengths('dog'), mi_event_length=mi_event_length, ma_event_length=ma_event_length)
baby_event_lengths = Distribution(load_event_lengths('baby'), mi_event_length=mi_event_length, ma_event_length=ma_event_length)

gamma_08_1 = Distribution(np.random.gamma(0.8, 1., 1000) + 0.5, mi_event_length=mi_event_length, ma_event_length=ma_event_length)
gamma_02_1 = Distribution(np.random.gamma(0.2, 1., 1000) + 0.5, mi_event_length=mi_event_length, ma_event_length=ma_event_length)


# normal change variance
normal_3_01 = Distribution(np.random.normal(3, 0.1, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)
normal_3_1 = Distribution(np.random.normal(3, 0.5, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)

# normal change mean
normal_05_01 = Distribution(np.random.normal(0.5, 0.1, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)
normal_5_01 = Distribution(np.random.normal(4, 0.1, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)

gammas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
#n_events = 30


parser = argparse.ArgumentParser(description='Plot the theoretical and simulated results for FIX weak labeling.')
parser.add_argument('--distribution', type=str, required=True, help='The distribution to use. Options: uniform, uniform_30, uniform_100, normal_variance, normal_mean, dog_and_baby, gamma')
parser.add_argument('--method', type=str, required=False, default='FIX', help='The method to use. Options: FIX, RND')
parser.add_argument('--n_samples', type=int, required=False, default=1000, help='The number of simulated audio recording samples')
args = parser.parse_args()
print(args)
print(args.distribution)

n_samples = args.n_samples

if args.distribution == 'normal_variance':
    n_eventss = [1, 1]
    distributions = [normal_3_01, normal_3_1]
    distribution_names = ['normal(3, 0.1)', 'normal(3, 1)']
elif args.distribution == 'normal_mean':
    n_eventss = [1, 1]
    distributions = [normal_05_01, normal_5_01]
    distribution_names = ['normal(0.5, 0.1)', 'normal(5, 0.1)']
elif args.distribution == 'dog_and_baby':
    n_eventss = [1, 1]
    distributions = [dog_event_lengths, baby_event_lengths]
    distribution_names = ['dog', 'baby']
elif args.distribution == 'gamma':
    n_eventss = [1, 1]
    distributions = [gamma_08_1, gamma_02_1]
    distribution_names = ['gamma(0.8, 1) + 0.5', 'gamma(0.2, 1) + 0.5']
elif args.distribution == 'uniform':
    n_eventss = [1]
    distributions = [uniform_1_1]
    distribution_names = ['d_e = 1']
elif args.distribution == 'uniform_10':
    n_eventss = [10]
    distributions = [uniform_1_1]
    distribution_names = ['d_e = 1']
elif args.distribution == 'uniform_30':
    n_eventss = [30]
    distributions = [uniform_1_1]
    distribution_names = ['d_e = 1']
elif args.distribution == 'uniform_50':
    n_eventss = [50]
    distributions = [uniform_1_1]
    distribution_names = ['d_e = 1']
elif args.distribution == 'uniform_100':
    n_eventss = [100]
    distributions = [uniform_1_1]
    distribution_names = ['d_e = 1']
else:
    raise ValueError(f"Unknown distribution: {args.distribution}")



print("----------------------------------------------------------")
print("Simulation parameters")
print("----------------------------------------------------------")
print("Recording length    : ", recording_length)
print("Event length limits : ", mi_event_length, ma_event_length)
print("Number of samples   : ", n_samples)
print("Number of events    : ", n_eventss)
print("Distributions       : ", args.distribution)
print("Method              : ", args.method)
print("")

method = args.method

# set figure size to 6.5 inches wide and 3.5 inches high
width = 6.5 * 1.5

if args.distribution == 'uniform' and np.mean(n_eventss) == 1:
    fig, ax = plt.subplots(1, 2, figsize=(width * 2/3, 3))
else:
    fig, ax = plt.subplots(1, 3, figsize=(width, 3))


def sample_events(n_events, distribution, recording_length):
    events = []
    for _ in range(n_events):
        event_length = distribution.sample()
        event_start = np.random.uniform(0, recording_length-event_length)
        event_end = event_start + event_length
        event = (event_start, event_end)
        events.append(event)

    # merge overlapping events
    events = merge_intervals(events)

    return events

#fig, ax = plt.subplots(1, 3, figsize=(12, 4))
considered_event_lengths_for_distribution = []
for idx_color, (distribution, name) in enumerate(zip(distributions, distribution_names)):
    n_events = n_eventss[idx_color]
    if n_events > 1:
        name = f"M={n_events}"

    considered_events = []

    # find the min and max event length in the distribution
    # this is needed to set the min and max of the query length search space
    for _ in range(1000):
        # sample the events
        events = sample_events(n_events, distribution, recording_length)
        considered_events.append(events)

    considered_events = np.concatenate(considered_events)
    considered_event_lengths = [e[1] - e[0] for e in considered_events]
    considered_event_lengths_for_distribution.append(considered_event_lengths)

    mi_event_length_distribution = np.min(considered_event_lengths)
    ma_event_length_distribution = np.max(considered_event_lengths)

    print("----------------------------------------------------------")
    print("Distribution                  : ", name)
    print("Min distribution event length : ", mi_event_length_distribution)
    print("Max distribution event length : ", ma_event_length_distribution)

    metric_maxs = []
    query_length_metric_maxs = []
    query_lengths = np.linspace(mi_event_length_distribution/500, ma_event_length_distribution*5, 200)

    scores = np.zeros((len(gammas), len(query_lengths), n_samples))

    for idx_gamma, gamma in enumerate(tqdm.tqdm(gammas)):
        for idx_query_length, query_length in enumerate(query_lengths):
            for idx_sample in range(n_samples):
                # sample the events
                events = sample_events(n_events, distribution, recording_length)

                if idx_sample == 0 and idx_gamma == 0 and idx_query_length == 0:
                    print("----------------------------------------------------------")
                    print("Gamma        : ", gamma)
                    print("Query length : ", query_length)
                    print("Events       : ", events)
                    print("#Events      : ", len(events))

                # split the recording into fixed equal length segments
                B = int(recording_length/query_length)+1

                segments = np.linspace(0, recording_length, B)
                segment_starts = segments[:-1]
                segment_ends = segments[1:]
                fix_query_segments = list(zip(segment_starts, segment_ends))

                events_tree = IntervalTree(Interval(e[0], e[1], e) for e in events)

                scores_for_overlapping_query_segments = []
                for q in fix_query_segments:
                    overlapping_events = events_tree[q[0]:q[1]]

                    #n_overlapping_events = len(overlapping_events)
                    #if n_overlapping_events > 1:
                    #    print("Number of overlapping events: ", n_overlapping_events)

                    # initially query segment does not have presence label
                    presence_event = False

                    # if no overlapping events, continue
                    if not overlapping_events:
                        continue
                    total_presence_event = 0

                    # loop over all overlapping events and accumulate the overlap
                    for interval in overlapping_events:
                        e = interval.data
                        event_start, event_end = e
                        query_start, query_end = q
                        o = defs.overlap(query_start, query_end, event_start, event_end)
                        total_presence_event += o
                        
                        d_e = event_end - event_start
                        if defs.annotator_criterion(e, q, gamma):
                            presence_event = True
                        
                    d_q = query_end - query_start

                    # model random label noise by flipping the label with probability rho
                    rho = 0.0
                    r = np.random.uniform(0, 1)
                    if r < rho:
                        presence_event = not presence_event

                    if presence_event:
                        # accuracy is the fraction of the query that is covered by the event
                        score = total_presence_event / d_q
                    else:
                        # accuracy is the fraction of the query that is not covered by the event
                        score = (d_q - total_presence_event) / d_q
                    scores_for_overlapping_query_segments.append(score)

                scores[idx_gamma, idx_query_length, idx_sample] = np.mean(scores_for_overlapping_query_segments)
    
    metric_maxs = np.mean(scores, axis=2).max(axis=1)
    query_length_metric_maxs = query_lengths[np.argmax(np.mean(scores, axis=2), axis=1)]

    if args.distribution == 'uniform' and n_events == 1:
        ax[0].plot(gammas, metric_maxs, label='simulated', color=colors[idx_color])
        ax[1].plot(gammas, query_length_metric_maxs, label='simulated', color=colors[idx_color])
        ax[1].plot(gammas, [thms.Q_max(gamma, 1) for gamma in gammas], label=r'$d_q^* (d_e=1)$', color='black', linestyle='--')
    else:
        ax[0].plot(gammas, metric_maxs, label='simulated', color=colors[idx_color])
        ax[1].plot(gammas, [thms.Q_max(gamma, np.mean(considered_event_lengths)) for gamma in gammas], label=r'$d_q^* (d_e = \mu)$'.format(np.mean(considered_event_lengths)), color=colors[idx_color], linestyle='--')
        ax[1].plot(gammas, query_length_metric_maxs, label='simulated', color=colors[idx_color])


        ax[2].hist(considered_event_lengths, bins=np.arange(0, 5, 0.1), color=colors[idx_color], label=name, alpha=1/len(distributions), density=True)

for idx_color, (distribution, name) in enumerate(zip(distributions, distribution_names)):

    event_lengths = considered_event_lengths_for_distribution[idx_color]
    event_lengths = np.random.choice(event_lengths, 10000) # slow if too many samples

    mi_event_length_distribution = np.min(event_lengths)
    ma_event_length_distribution = np.max(event_lengths)

    print("----------------------------------------------------------")
    print("Distribution                  : ", name)
    print("Min distribution event length : ", mi_event_length_distribution)
    print("Max distribution event length : ", ma_event_length_distribution)
    print("Number of event lengths       : ", len(event_lengths))
    
    
    #gammas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    query_lengths = np.linspace(mi_event_length_distribution/500, ma_event_length_distribution*5, 200)

    max_query_scores = []
    max_query_lengths = []
    for gamma in tqdm.tqdm(gammas):
        query_scores = []
        for query_length in query_lengths:
            f = partial(thms.P, d_q=query_length, gamma=gamma)
            xs = event_lengths
            ys = [f(x) for x in xs]
            average = np.mean(ys)
            query_scores.append(average)

        idx = np.argmax(query_scores)
        max_query_length = query_lengths[idx]
        max_query_score = query_scores[idx]

        max_query_scores.append(max_query_score)
        max_query_lengths.append(max_query_length)

    if not (args.distribution == 'uniform' and n_events == 1):
        ax[0].plot(gammas, max_query_scores, label='numerical', color=colors[idx_color], linestyle='', marker='x')
        ax[1].plot(gammas, max_query_lengths, label='numerical', color=colors[idx_color], linestyle='', marker='x')

ax[0].plot(gammas, [thms.P_max(gamma) for gamma in gammas], label=r'$f^*(\gamma)$', color='black', linestyle='--')
ax[0].set_xlabel(r'$\gamma$')
ax[0].set_ylabel('Maximum expected label accuracy')

# create a sorted legend
handles, labels = ax[0].get_legend_handles_labels()
handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: x[1])
handles_sorted, labels_sorted = zip(*handles_labels_sorted)
ax[0].legend(handles_sorted, labels_sorted)
ax[0].set_ylim(0, 1)
ax[0].set_xlim(0, 1)

event_length_averages = []
for distribution in distributions:
    print("Min: ", distribution.min())
    print("Max: ", distribution.max())
    event_length_averages.append(np.mean(distribution.event_lengths))
ma = np.max(event_length_averages)

ax[1].set_xlabel(r'$\gamma$')
ax[1].set_ylabel('Optimal query length')

# create a sorted legend
handles, labels = ax[1].get_legend_handles_labels()
handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: x[1])
handles_sorted, labels_sorted = zip(*handles_labels_sorted)
ax[1].legend(handles_sorted, labels_sorted)

ax[1].set_ylim(0, ma*4)
ax[1].set_xlim(0, 1)

if not (args.distribution == 'uniform' and n_events == 1):
    ax[2].set_xlabel(r"$d_e$")
    ax[2].set_ylabel(r"$p(d_e)$")
    # create a sorted legend
    handles, labels = ax[2].get_legend_handles_labels()
    handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: x[1])
    handles_sorted, labels_sorted = zip(*handles_labels_sorted)
    ax[2].legend(handles_sorted, labels_sorted)

if not os.path.exists("figures"):
    os.makedirs("figures")

plt.tight_layout()
plt.savefig(f"figures/{args.distribution}_{args.method}.png")
