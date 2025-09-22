#!/usr/bin/env python3
import os
import definitions as defs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import definitions as defs
import numpy as np


def get_times_and_metric_scores(d_e, d_q, gamma, metric):
        metric_func = lambda t: defs.occurences_with_overlap(t, d_e, d_q, gamma, metric)
        # plot
        ys = []
        xs = np.linspace(0, d_e + d_q, 1000)

        for t in xs:
            y = metric_func(t)
            ys.append(y)
        
        return xs, ys

def get_ts(d_e, d_q, gamma):
    t_0 = 0
    t_1 = gamma * d_e
    if d_e >= d_q:
        t_2 = d_q
        t_3 = d_e
    else:
        t_2 = d_e
        t_3 = d_q
    
    t_4 = t_3 + (t_2-t_1)
    t_5 = d_e + d_q

    return t_0, t_1, t_2, t_3, t_4, t_5

def get_fs(d_e, d_q, gamma):
    f_0  = 1
    f_11 = (d_q - gamma*d_e)/d_q
    f_12 = gamma*d_e/d_q
    if d_e >= d_q:
        f_2  = 1
    else:
        f_2  = d_e/d_q

    if d_q <= gamma*d_e:
        f_11 = 0
        f_12 = 0
        f_2  = 0
    
    return f_0, f_11, f_12, f_2


def metric_at_all_overlap_occurences(ax):
    metric = defs.query_segment_accuracy_score

    # Plot the metric as a function of t for the two cases (i) d_e >= d_q and (ii) d_e < d_q
    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    gamma = 0.5

    # assumption d_q >= gamma * d_e
    # case (i), d_e >= d_q >= gamma*d_e
    d_e_i = 1.5
    d_q_i = 1
    xs_i, ys_i = get_times_and_metric_scores(d_e=d_e_i, d_q=d_q_i, gamma=gamma, metric=metric)
    t_0_i, t_1_i, t_2_i, t_3_i, t_4_i, t_5_i = get_ts(d_e=d_e_i, d_q=d_q_i, gamma=gamma)
    f_0_i, f_11_i, f_12_i, f_2_i = get_fs(d_e=d_e_i, d_q=d_q_i, gamma=gamma)

    # case (ii), d_e < d_q
    d_e_ii = 1
    d_q_ii = 1.5
    xs_ii, ys_ii = get_times_and_metric_scores(d_e=d_e_ii, d_q=d_q_ii, gamma=gamma, metric=metric)
    t_0_ii, t_1_ii, t_2_ii, t_3_ii, t_4_ii, t_5_ii = get_ts(d_e=d_e_ii, d_q=d_q_ii, gamma=gamma)
    f_0_ii, f_11_ii, f_12_ii, f_2_ii = get_fs(d_e=d_e_ii, d_q=d_q_ii, gamma=gamma)

    # assumption d_q < gamma * d_e
    d_e_iii = 1.5
    d_q_iii = 0.5 # gamma * d_e_iii - 1e-6
    xs_iii, ys_iii = get_times_and_metric_scores(d_e=d_e_iii, d_q=d_q_iii, gamma=gamma, metric=metric)
    t_0_iii, t_1_iii, t_2_iii, t_3_iii, t_4_iii, t_5_iii = get_ts(d_e=d_e_iii, d_q=d_q_iii, gamma=gamma)
    f_0_iii, f_11_iii, f_12_iii, f_2_iii = get_fs(d_e=d_e_iii, d_q=d_q_iii, gamma=gamma)



    # plot a vertical dashed line at t_0_i, t_1_i, t_2_i, t_3_i, t_4_i, t_5_i
    ax[0].plot([t_0_i, t_0_i], [0, f_0_i], linestyle="--", color="gray")
    ax[0].plot([t_1_i, t_1_i], [0, f_11_i], linestyle="--", color="gray")
    ax[0].plot([t_2_i, t_2_i], [0, f_2_i], linestyle="--", color="gray")
    ax[0].plot([t_3_i, t_3_i], [0, f_2_i], linestyle="--", color="gray")
    ax[0].plot([t_4_i, t_4_i], [0, f_11_i], linestyle="--", color="gray")
    ax[0].plot([t_5_i, t_5_i], [0, f_0_i], linestyle="--", color="gray")

    # plot a vertical dashed line at t_0_ii, t_1_ii, t_2_ii, t_3_ii, t_4_ii, t_5_ii
    ax[1].plot([t_0_ii, t_0_ii], [0, f_0_ii], linestyle="--", color="gray")
    ax[1].plot([t_1_ii, t_1_ii], [0, f_11_ii], linestyle="--", color="gray")
    ax[1].plot([t_2_ii, t_2_ii], [0, f_2_ii], linestyle="--", color="gray")
    ax[1].plot([t_3_ii, t_3_ii], [0, f_2_ii], linestyle="--", color="gray")
    ax[1].plot([t_4_ii, t_4_ii], [0, f_11_ii], linestyle="--", color="gray")
    ax[1].plot([t_5_ii, t_5_ii], [0, f_0_ii], linestyle="--", color="gray")

    # plot a vertical dashed line at t_0_iii, t_1_iii, t_2_iii, t_3_iii, t_4_iii, t_5_iii
    ax[2].plot([t_0_iii, t_0_iii], [0, f_0_iii], linestyle="--", color="gray")
    #ax[2].plot([t_1_iii, t_1_iii], [0, f_11_iii], linestyle="--", color="gray")
    ax[2].plot([t_2_iii, t_2_iii], [0, f_2_iii], linestyle="--", color="gray")
    ax[2].plot([t_3_iii, t_3_iii], [0, f_2_iii], linestyle="--", color="gray")
    #ax[2].plot([t_4_iii, t_4_iii], [0, f_11_iii], linestyle="--", color="gray")
    ax[2].plot([t_5_iii, t_5_iii], [0, f_0_iii], linestyle="--", color="gray")


    # add custom xticks and xticklabels at t_0_i, t_1_i, t_2_i, t_3_i, t_4_i, t_5_i
    xticks_i = [t_0_i, t_1_i, t_2_i, t_3_i, t_4_i, t_5_i]
    xticklabels_i = [r"$t^{(i)}_0$", r"$t^{(i)}_1$", r"$t^{(i)}_2$", r"$t^{(i)}_3$", r"$t^{(i)}_4$", r"$t^{(i)}_5$"]
    ax[0].set_xticks(xticks_i)
    ax[0].set_xticklabels(xticklabels_i)


    # add custom xticks and xticklabels at t_0_ii, t_1_ii, t_2_ii, t_3_ii, t_4_ii, t_5_ii
    xticks_ii = [t_0_ii, t_1_ii, t_2_ii, t_3_ii, t_4_ii, t_5_ii]
    xticklabels_ii = [r"$t^{(ii)}_0$", r"$t^{(ii)}_1$", r"$t^{(ii)}_2$", r"$t^{(ii)}_3$", r"$t^{(ii)}_4$", r"$t^{(ii)}_5$"]
    ax[1].set_xticks(xticks_ii)
    ax[1].set_xticklabels(xticklabels_ii)

    # add custom xticks and xticklabels at t_0_iii, t_1_iii, t_2_iii, t_3_iii, t_4_iii, t_5_iii
    xticks_iii = [t_0_iii, t_2_iii, t_3_iii, t_5_iii]
    xticklabels_iii = [r"$t_0$", r"$t_1$", r"$t_2$", r"$t_3$"]
    ax[2].set_xticks(xticks_iii)
    ax[2].set_xticklabels(xticklabels_iii)

    ax[0].plot(xs_i, ys_i)
    ax[1].plot(xs_ii, ys_ii)
    ax[2].plot(xs_iii, ys_iii)

    # mark the area under the curve ys_i between x=t_0_i and x=t_1_i with dashed lines '//'
    ax[0].fill_between(xs_i, ys_i, where=(xs_i >= t_0_i) & (xs_i <= t_1_i), color="#D3D3D3", alpha=0.5)#, hatch="//")
    ax[0].fill_between(xs_i, ys_i, where=(xs_i >= t_1_i) & (xs_i <= t_2_i), color="#D3D3D3", alpha=0.5)#, hatch="\\\\")
    ax[0].fill_between(xs_i, ys_i, where=(xs_i >= t_2_i) & (xs_i <= t_3_i), color="#D3D3D3", alpha=0.5)
    ax[0].fill_between(xs_i, ys_i, where=(xs_i >= t_3_i) & (xs_i <= t_4_i), color="#D3D3D3", alpha=0.5)#, hatch="\\\\")
    ax[0].fill_between(xs_i, ys_i, where=(xs_i >= t_4_i) & (xs_i <= t_5_i), color="#D3D3D3", alpha=0.5)#, hatch="//")

    # mark the area under the curve ys_ii between x=t_0_ii and x=t_1_ii with dashed lines '//'
    ax[1].fill_between(xs_ii, ys_ii, where=(xs_ii >= t_0_ii) & (xs_ii <= t_1_ii), color="#D3D3D3", alpha=0.5)#, hatch="//")
    ax[1].fill_between(xs_ii, ys_ii, where=(xs_ii >= t_1_ii) & (xs_ii <= t_2_ii), color="#D3D3D3", alpha=0.5)#, hatch="\\\\")
    ax[1].fill_between(xs_ii, ys_ii, where=(xs_ii >= t_2_ii) & (xs_ii <= t_3_ii), color="#D3D3D3", alpha=0.5)
    ax[1].fill_between(xs_ii, ys_ii, where=(xs_ii >= t_3_ii) & (xs_ii <= t_4_ii), color="#D3D3D3", alpha=0.5)#, hatch="\\\\")
    ax[1].fill_between(xs_ii, ys_ii, where=(xs_ii >= t_4_ii) & (xs_ii <= t_5_ii), color="#D3D3D3", alpha=0.5)#, hatch="//")

    # mark the area under the curve ys_iii between x=t_0_iii and x=t_1_iii with dashed lines '//'
    ax[2].fill_between(xs_iii, ys_iii, where=(xs_iii >= t_0_iii) & (xs_iii <= t_1_iii), color="#D3D3D3", alpha=0.5)#, hatch="//")
    ax[2].fill_between(xs_iii, ys_iii, where=(xs_iii >= t_1_iii) & (xs_iii <= t_4_iii), color="#D3D3D3", alpha=0.5)#, hatch="\\\\")
    ax[2].fill_between(xs_iii, ys_iii, where=(xs_iii >= t_4_iii) & (xs_iii <= t_5_iii), color="#D3D3D3", alpha=0.5)#, hatch="//")


    # plot a text A_1, A_2, A_3, A_2, A_1 at the center of the areas for case (i)
    ax[0].text((t_0_i + t_1_i)/2, 0.3, r"$A^{(i)}_1$", fontsize=12, ha="center")
    ax[0].text((t_1_i + t_2_i)/2, 0.3, r"$A^{(i)}_2$", fontsize=12, ha="center")
    ax[0].text((t_2_i + t_3_i)/2, 0.3, r"$A^{(i)}_3$", fontsize=12, ha="center")
    ax[0].text((t_3_i + t_4_i)/2, 0.3, r"$A^{(i)}_2$", fontsize=12, ha="center")
    ax[0].text((t_4_i + t_5_i)/2, 0.3, r"$A^{(i)}_1$", fontsize=12, ha="center")

    # plot a text A_1, A_2, A_3, A_2, A_1 at the center of the areas for case (ii)
    ax[1].text((t_0_ii + t_1_ii)/2, 0.3, r"$A^{(ii)}_1$", fontsize=12, ha="center")
    ax[1].text((t_1_ii + t_2_ii)/2, 0.3, r"$A^{(ii)}_2$", fontsize=12, ha="center")
    ax[1].text((t_2_ii + t_3_ii)/2, 0.3, r"$A^{(ii)}_3$", fontsize=12, ha="center")
    ax[1].text((t_3_ii + t_4_ii)/2, 0.3, r"$A^{(ii)}_2$", fontsize=12, ha="center")
    ax[1].text((t_4_ii + t_5_ii)/2, 0.3, r"$A^{(ii)}_1$", fontsize=12, ha="center")

    # plot a text A_1, A_1 at the center of the areas for case (iii)
    offset = 0.07
    ax[2].text((t_0_iii + t_2_iii)/2 - offset, 0.3, r"$A_1$", fontsize=12, ha="center")
    ax[2].text((t_3_iii + t_5_iii)/2 + offset, 0.3, r"$A_1$", fontsize=12, ha="center")



    

    # plot settings
    ax[0].set_title(r"Case (i): $d_e \geq d_q$")
    ax[0].set_xlabel(r"$t$")
    ax[0].set_ylabel(r"$F(e_t, q, \gamma)$")
    ax[0].set_ylim(0, 1.1)
    ax[0].set_xlim(0, d_e_i + d_q_i)

    ax[1].set_title(r"Case (ii): $d_e < d_q$")
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$F(e_t, q, \gamma)$")
    ax[1].set_ylim(0, 1.1)
    ax[1].set_xlim(0, d_e_ii + d_q_ii)

    ax[2].set_title(r"$d_q < \gamma d_e$")
    ax[2].set_xlabel(r"$t$")
    ax[2].set_ylabel(r"$F(e_t, q, \gamma)$")
    ax[2].set_ylim(0, 1.1)
    ax[2].set_xlim(0, d_e_iii + d_q_iii)

def main():

    if not os.path.exists('figures'):
        os.makedirs('figures')

    #############################################################
    # Figure 15 and 16
    #############################################################
    # define the figsize for a 1 column journal paper
    fig1 = plt.figure(figsize=(2*3.5, 2.5))
    fig2 = plt.figure(figsize=(3.5, 2.5))
    # Create a 2x2 GridSpec
    gs1 = gridspec.GridSpec(1, 2)
    gs2 = gridspec.GridSpec(1, 1)

    # Plot the two upper figures
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax2 = fig1.add_subplot(gs1[0, 1])

    # Plot the centered bottom figure
    ax3 = fig2.add_subplot(gs2[0, 0])

    ax = [ax1, ax2, ax3]

    metric_at_all_overlap_occurences(ax=ax)
    fig1.tight_layout()
    # figure 15
    fig1.savefig("figures/metric_at_all_overlap_occurences_1.png")
    fig2.tight_layout()
    # figure 16
    fig2.savefig("figures/metric_at_all_overlap_occurences_2.png")
    #############################################################

    #############################################################
    # Figure 4
    #############################################################
    figsize = (2*3.5, 2.5)
    fig = plt.figure(figsize=figsize)

    # Create a GridSpec layout with 3 rows and 1 column
    # The height ratios are [1, 1, 2] to make the bottom plot twice the height of the top plots
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    #ax3 = fig.add_subplot(gs[2, 0])

    ax = [ax1, ax2] #, ax3]

    T = 30
    B = 7
    d_e = 2
    d_q = T/B
    gamma = 0.5
    q_idx = 2

    qs = [[i*d_q, (i+1)*d_q] for i in range(int(T/d_q))]

    metric = defs.query_segment_accuracy_score

    def occurences_e(t, d_e, T, q, gamma, metric):
        """
        Evaluate the metric for a given t by defining the event timings relative to the query timings.
        """
        offset = -d_e
        e  = [t + offset, d_e+t+offset]

        return metric(e, q, gamma)

    metric_t = lambda t: occurences_e(t, d_e, T, qs[q_idx], gamma, metric)

    n_samples = 10000
    # qs is a list of query segments [[q1_onset, q1_end], [q2_start, q2_end], ...]
    # plot a dashed vertical red line at the start of the each query segment
    for q in qs:
        ax[0].axvline(x=q[0], color='r', linestyle='--')

    ax[0].text((qs[q_idx][0] + qs[q_idx][1])/2, 0.5, r"$q_{}$".format(q_idx), fontsize=12, ha='center')
    xs = np.linspace(0, T, n_samples)
    ys = [metric_t(t) for t in xs]
    ax[1].plot(xs, ys)

    # fill the area above the curve ys, at x=qs[q_idx]-d_e/2 and x=qs[q_idx]+d_e/2 with a red color
    x_start = qs[q_idx][0] # - d_e
    x_end = qs[q_idx][1] + d_e
    #ax[1].fill_between(xs, ys, 1, where=(xs >= x_start) & (xs <= x_end), color='red', alpha=0.2)
    ax[1].fill_between(xs, 0, ys, where=(xs >= x_start) & (xs <= x_end), color='red', alpha=0.1, hatch='//')


    # plot the event segment in ax[0] with arrows in both directions to indicate that it can appear anywhere in the time series
    # the start time of the event segment is at the center of the plot

    # plot the event as a green area in ax[0]
    ax[0].fill_between([d_e/2 - d_e/2, d_e/2 + d_e/2], 0, 1.1, color='g', alpha=0.5)

    ax[1].fill_between(xs, 0, 1, where=(xs >= 0) & (xs <= x_start), color='green', alpha=0.1, hatch='//')
    ax[1].fill_between(xs, 0, 1, where=(xs >= x_end) & (xs <= T), color='green', alpha=0.1, hatch='//')

    ax[0].text(d_e/2, 0.5, r'$e_{t=0}$', fontsize=12, ha='center')
    # the arrows are of length d_q and start at an offset of d_e/2 from the center of the plot
    #ax[0].arrow(d_e - d_e/2, 0.5, -d_q/2, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')
    ax[0].arrow(d_e/2 + d_e/2, 0.5, d_q, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')

    # plot a line |----------------| at the bottom of the plot between x_start and x_end
    #ax[1].plot([x_start, x_end], [0.2, 0.2], color='k')

    # Annotate the line below the plot
    #ax[1].annotate('', xy=(x_start-0.3, 0.05), xytext=(x_end+0.3, 0.05),
    #               arrowprops=dict(arrowstyle='<->', color='k'))
    ax[1].text((x_start + x_end) / 2, -0.30, r'$d_e + d_q$', fontsize=12, ha='center')
    ax[1].text((x_start + x_end) / 2, 0.20, r'A', fontsize=12, ha='center')

    # Add the text labels
    loc = 1.2
    ax[0].text(d_e / 2, loc, r'$d_e$', ha='center', va='bottom')
    ax[0].text((qs[q_idx][0] + qs[q_idx][1]) / 2, loc, r'$d_q$', ha='center', va='bottom')

    # Plot lines below the text labels
    line_y = loc - 0.1  # Adjust this value for vertical spacing
    tick_length = 0.02

    for idx, ax in enumerate(ax):
        if idx in [0]:
            ax.set_yticks([])
        if idx == 1:
            ax.set_xlabel("t")
            # set custom xticks labels so that T is shown at 30
            ax.set_xticks([0, 30])
            ax.set_xticklabels(['0', 'T'])
        else:
            ax.set_xticks([])
        if idx in [1]:
            ax.set_ylabel(r"$F(e_t, q_{}, \gamma)$".format(q_idx))
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, T])

    fig.tight_layout()
    fig.savefig(f"figures/occurences_{q_idx}.png")
    #############################################################

    #############################################################
    # Figure 5
    #############################################################
    # Define the figure size
    figsize = (2*3.5, 2.5)
    fig = plt.figure(figsize=figsize)

    # Create a GridSpec layout with 3 rows and 1 column
    # The height ratios are [1, 1, 2] to make the bottom plot twice the height of the top plots
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax = [ax1, ax2]

    T = 30
    B = 7
    d_e = 2
    d_q = T/B
    gamma = 0.5
    q_idx = 2

    offset_event_1 = 10
    offset_event_2 = 20

    e1 = np.array([offset_event_1, offset_event_1+d_e]).astype(float)
    e2 = np.array([offset_event_2, offset_event_2+d_e]).astype(float)
    es = [e1, e2]

    qs = [[i*d_q, (i+1)*d_q] for i in range(int(T/d_q))]
    q = [0, d_q]

    metric = defs.query_segment_accuracy_score

    # move the query segment instead of the event segment
    def occurences_q(t, d_q, T, es, gamma, metric):
        """
        Evaluate the metric for a given t by defining the query timings relative to the event timings.
        """
        q  = [t, d_q+t]

        accuracies = []
        for e in es:
            accuracy = metric(e, q, gamma)
            accuracies.append(accuracy)
        accuracy = np.min(accuracies)
        #print("accuracy: ", accuracy)

        return accuracy
    
    metric_t_es = lambda t: occurences_q(t, d_q, T, es, gamma, metric)

    n_samples = 10000
    # qs is a list of query segments [[q1_onset, q1_end], [q2_start, q2_end], ...]
    # plot a dashed vertical red line at the start of the each query segment
    #for q in qs:
    ax[0].axvline(x=q[0]+0.1, color='r', linestyle='--')
    ax[0].axvline(x=q[1], color='r', linestyle='--')

    #ax[0].text((qs[q_idx][0] + qs[q_idx][1])/2, 0.5, r"$q_{}$".format(q_idx), fontsize=12, ha='center')
    xs = np.linspace(0, T, n_samples)
    ys = [metric_t_es(t) for t in xs]
    ax[1].plot(xs, ys)

    # fill the area above the curve ys, for event 1
    x_start_1 = e1[0] - d_q
    x_end_1 = e1[1]
    #ax[1].fill_between(xs, ys, 1, where=(xs >= x_start_1) & (xs <= x_end_1), color='red', alpha=0.2)
    ax[1].fill_between(xs, 0, ys, where=(xs >= x_start_1) & (xs <= x_end_1), color='red', alpha=0.1, hatch='//')
    ax[1].text((x_start_1 + x_end_1) / 2, -0.30, r'$d_e + d_q$', fontsize=12, ha='center')
    ax[1].text((x_start_1 + x_end_1) / 2, 0.20, r'$A$', fontsize=12, ha='center')

    # fill the area above the curve ys, for event 2
    x_start_2 = e2[0] - d_q
    x_end_2 = e2[1]
    #ax[1].fill_between(xs, ys, 1, where=(xs >= x_start_2) & (xs <= x_end_2), color='red', alpha=0.2)
    ax[1].fill_between(xs, 0, ys, where=(xs >= x_start_2) & (xs <= x_end_2), color='red', alpha=0.1, hatch='//')
    ax[1].text((x_start_2 + x_end_2) / 2, -0.30, r'$d_e + d_q$', fontsize=12, ha='center')
    ax[1].text((x_start_2 + x_end_2) / 2, 0.20, r'$A$', fontsize=12, ha='center')

    # plot the event as a green area in ax[0]
    ax[0].fill_between(e1, 0, 1.1, color='g', alpha=0.5)
    ax[0].fill_between(e2, 0, 1.1, color='g', alpha=0.5)
    ax[0].fill_between(q, 0, 1.1, color='r', alpha=0.3)

    ax[0].text(np.mean(e1), 0.5, r'$e_1$', fontsize=12, ha='center')
    ax[0].text(np.mean(e2), 0.5, r'$e_2$', fontsize=12, ha='center')
    ax[0].text(np.mean(q), 0.5, r'$q_{t=0}$', fontsize=12, ha='center')

    ax[0].arrow(q[1], 0.5, 1, 0, head_width=0.2, head_length=0.3, fc='k', ec='k')

    # fill the areas where always accuracy 1, which is between 0 and e1[0]-d_q, e1[1] and e2[0]-d_q, e2[1] and T
    ax[1].fill_between(xs, 0, 1, where=(xs >= 0) & (xs <= e1[0]-d_q), color='green', alpha=0.1, hatch='//')
    ax[1].fill_between(xs, 0, 1, where=(xs >= e1[1]) & (xs <= e2[0]-d_q), color='green', alpha=0.1, hatch='//')
    ax[1].fill_between(xs, 0, 1, where=(xs >= e2[1]) & (xs <= T), color='green', alpha=0.1, hatch='//')


    # Add the text labels
    loc = 1.2
    ax[0].text(np.mean(e1), loc, r'$d_e$', ha='center', va='bottom')
    ax[0].text(np.mean(e2), loc, r'$d_e$', ha='center', va='bottom')
    ax[0].text(np.mean(q), loc, r'$d_q$', ha='center', va='bottom')

    # Plot lines below the text labels
    line_y = loc - 0.1  # Adjust this value for vertical spacing
    tick_length = 0.02

    for idx, ax in enumerate(ax):
        if idx in [0]:
            ax.set_yticks([])
        if idx == 1:
            ax.set_xlabel("t")
            # set custom xticks labels so that T is shown at 30
            ax.set_xticks([0, 30])
            ax.set_xticklabels(['0', 'T'])
        else:
            ax.set_xticks([])
        if idx in [1]:
            ax.set_ylabel(r"$F(e, q_t, \gamma)$".format(q_idx))
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, T])

    fig.tight_layout()
    fig.savefig(f"figures/occurences_multi_events.png")
    #############################################################

if __name__ == "__main__":
    main()
