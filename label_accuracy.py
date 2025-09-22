import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glob
# import audio_utils.annotations as aua
# import audio_utils.stats as aus
import sys
from functools import partial
# from intervaltree import Interval, IntervalTree

import argparse


from scipy.stats import gamma as gamma_pdf

import definitions as defs
import theorems as thms


# def theoretical_max_iou_rule(gamma):
#     return 1 + 2*gamma**2 - 2*np.sqrt(gamma**2 + gamma**4)

# def theoretical_length_rule(gamma, event_length):
#         return np.sqrt(event_length**2 * (gamma**2 + gamma**4)) + event_length*gamma**2

# def load_event_lengths(class_name):
#     dir_path = f"/home/john/gits/adaptive-change-point-detection-extended/data/generated_datasets/{class_name}_1.0_0.25s/train_soundscapes_snr_0.0"


#     # load the annotations
#     annotation_files = glob.glob(f"{dir_path}/*.csv")
#     annotation_files = [file for file in annotation_files if 'embeddings' not in file]
#     annotations = aua.load_annotation_files(annotation_files)

#     # load the event lengths
#     event_lengths = aus.compute_event_lengths(annotations)
#     return event_lengths


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

recording_length = 3
mi_event_length = 0.01
ma_event_length = recording_length
    
# different distributions
uniform_1_1 = Distribution(np.random.uniform(1, 1, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)
uniform_02_02 = Distribution(np.random.uniform(0.2, 0.2, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)

#dog_event_lengths = Distribution(load_event_lengths('dog'), mi_event_length=mi_event_length, ma_event_length=ma_event_length)
#baby_event_lengths = Distribution(load_event_lengths('baby'), mi_event_length=mi_event_length, ma_event_length=ma_event_length)

gamma_08_1 = Distribution(np.random.gamma(0.8, 1., 1000) + 0.5, mi_event_length=mi_event_length, ma_event_length=ma_event_length)
gamma_02_1 = Distribution(np.random.gamma(0.2, 1., 1000) + 0.5, mi_event_length=mi_event_length, ma_event_length=ma_event_length)


# normal change variance
normal_3_01 = Distribution(np.random.normal(3, 0.1, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)
normal_3_1 = Distribution(np.random.normal(3, 0.5, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)

# normal change mean
normal_05_01 = Distribution(np.random.normal(0.5, 0.1, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)
#normal_5_01 = Distribution(np.random.normal(4, 0.1, 10000), mi_event_length=mi_event_length, ma_event_length=ma_event_length)

n_samples = 100
gammas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
#n_events = 30


parser = argparse.ArgumentParser(description='Plot the theoretical and simulated results for FIX weak labeling.')
parser.add_argument('--distribution', type=str, required=True, help='The distribution to use. Options: uniform, uniform_30, uniform_100, normal_variance, normal_mean, dog_and_baby, gamma')
parser.add_argument('--method', type=str, required=False, default='FIX', help='The method to use. Options: FIX, RND')
parser.add_argument('--metric', type=str, required=False, default='accuracy', help='The metric to use. Options: iou, f1, accuracy')
args = parser.parse_args()
print(args)
print(args.distribution)


if args.distribution == 'normal_variance':
    n_eventss = [1, 1]
    distributions = [normal_3_01, normal_3_1]
    distribution_names = ['normal(3, 0.1)', 'normal(3, 1)']
elif args.distribution == 'normal_mean':
    n_eventss = [1] #, 1]
    distributions = [normal_05_01] #, normal_5_01]
    distribution_names = ['normal(0.5, 0.1)'] #, 'normal(5, 0.1)']
# elif args.distribution == 'dog_and_baby':
#     n_eventss = [1, 1]
#     distributions = [dog_event_lengths, baby_event_lengths]
#     distribution_names = ['dog', 'baby']
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
print("Metric              : ", args.metric)
print("")

method = args.method
if args.metric == 'iou':
    metric_fn = defs.iou_score
elif args.metric == 'f1':
    metric_fn = defs.f1_score
elif args.metric == 'accuracy':
    metric_fn = defs.query_segment_accuracy_score
else:
    raise ValueError(f"Unknown metric: {args.metric}")

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
    print("Number of events              : ", n_events)

    metric_maxs = []
    query_length_metric_maxs = []
    query_lengths = np.linspace(mi_event_length_distribution/500, ma_event_length_distribution*3, 200)

    scores = np.zeros((len(gammas), len(query_lengths), n_samples))

    for idx_gamma, gamma in enumerate(tqdm.tqdm(gammas)):
        for idx_query_length, query_length in enumerate(query_lengths):
            for idx_sample in range(n_samples):
                # sample the events
                events = sample_events(n_events, distribution, recording_length)

                # split the recording into fixed equal length segments
                B = int(recording_length/query_length)+1

                segments = np.linspace(0, recording_length, B)
                segment_starts = segments[:-1]
                segment_ends = segments[1:]
                fix_query_segments = list(zip(segment_starts, segment_ends))
                
                # assert that the end of the last segment is equal to the recording length
                assert fix_query_segments[-1][1] == recording_length
                # assert that all segments except the last one have length query_length
                #assert all([q[1] - q[0] == query_length for q in fix_query_segments[:-1]])
                
                # simulate weak labeling of the query segments with respect to the events
                # using the annotator criterion
                weak_segment_labels = []
                for q in fix_query_segments:
                    presence_event = False
                    for e in events:
                        if defs.annotator_criterion(e, q, gamma):
                            presence_event = True
                    weak_segment_labels.append(presence_event)

                # merge adjacent query segments with the same weak label
                presence_timings = [(q[0], q[1]) for q, p in zip(fix_query_segments, weak_segment_labels) if p]
                absence_timings  = [(q[0], q[1]) for q, p in zip(fix_query_segments, weak_segment_labels) if not p]
                presence_labels  = merge_intervals(presence_timings)
                absence_labels   = merge_intervals(absence_timings)

                # compute the accuracy of the presence_labels with respect to the events for the full recording
                # simply loop over some time points and compute the accuracy for each time point by checking if
                # the time point is in the presence_labels AND the events, or if the time point is not in the presence_labels AND not in the events
                n_points = 1000
                agreement = 0
                for t in np.linspace(0, recording_length, n_points):

                    is_in_presence_labels = False
                    is_in_events = False
                    for e in events:
                        if e[0] <= t <= e[1]:
                            is_in_events = True
                    for p in presence_labels:
                        if p[0] <= t <= p[1]:
                            is_in_presence_labels = True
                    
                    agreement += is_in_presence_labels == is_in_events
                accuracy = agreement / n_points

                scores[idx_gamma, idx_query_length, idx_sample] = accuracy
    
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


# def theoretical_iou_rule(event_length, query_length, gamma):
#     if query_length >= gamma*event_length:
#         return (1/(event_length + query_length)) * (event_length - (event_length**2 * gamma**2)/query_length)
#     else:
#         return 0
    

for idx_color, (distribution, name) in enumerate(zip(distributions, distribution_names)):
    n_events = n_eventss[idx_color]

    event_lengths = considered_event_lengths_for_distribution[idx_color]
    event_lengths = np.random.choice(event_lengths, 10000) # slow if too many samples

    mi_event_length_distribution = np.min(event_lengths)
    ma_event_length_distribution = np.max(event_lengths)

    print("----------------------------------------------------------")
    print("Distribution                  : ", name)
    print("Min distribution event length : ", mi_event_length_distribution)
    print("Max distribution event length : ", ma_event_length_distribution)
    print("Number of events              : ", n_events)
    print("Number of event lengths       : ", len(event_lengths))
    
    
    #gammas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    query_lengths = np.linspace(mi_event_length_distribution/500, ma_event_length_distribution*3, 200)

    max_query_scores = []
    max_query_lengths = []
    for gamma in tqdm.tqdm(gammas):
        query_scores = []
        for query_length in query_lengths:
            #f = partial(thms.P, d_q=query_length, gamma=gamma)
            f = partial(thms.TheoreticalLabelAccuracy, d_q=query_length, gamma=gamma, M=n_events, T=recording_length)
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
        print("Max query length: ", max_query_lengths)
        print("Max query score: ", max_query_scores)
        ax[0].plot(gammas, max_query_scores, label='numerical', color=colors[idx_color], linestyle='', marker='x')
        ax[1].plot(gammas, max_query_lengths, label='numerical', color=colors[idx_color], linestyle='', marker='x')

#accuracies = [thms.P_max(gamma) for gamma in gammas]
accuracies = [thms.TheoreticalLabelAccuracyMax(d_e=0.5, gamma=gamma, M=1, T=recording_length) for gamma in gammas]
ax[0].plot(gammas, accuracies, label=r'$f^*(\gamma)$', color='black', linestyle='--')
ax[0].set_xlabel(r'$\gamma$')
ax[0].set_ylabel('Maximum expected query accuracy')

# create a sorted legend
handles, labels = ax[0].get_legend_handles_labels()
handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: x[1])
handles_sorted, labels_sorted = zip(*handles_labels_sorted)
ax[0].legend(handles_sorted, labels_sorted)
#ax[0].set_ylim(0, 1)
#ax[0].set_xlim(0, 1)

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

#ax[1].set_ylim(0, ma*4)
#ax[1].set_xlim(0, 1)

if not (args.distribution == 'uniform' and n_events == 1):
    ax[2].set_xlabel(r"$d_e$")
    ax[2].set_ylabel(r"$p(d_e)$")
    # create a sorted legend
    handles, labels = ax[2].get_legend_handles_labels()
    handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: x[1])
    handles_sorted, labels_sorted = zip(*handles_labels_sorted)
    ax[2].legend(handles_sorted, labels_sorted)

plt.tight_layout()
plt.savefig(f"figures/{args.distribution}_{args.method}_{args.metric}.png")