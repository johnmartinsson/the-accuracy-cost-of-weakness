import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

import definitions as defs

# length of e
#d_e = 1
d_e = 1.3
# length of q
d_q = 1.2
gamma = 0.5 #0.5

if d_q < gamma*d_e:
    raise ValueError("d_q must be greater than gamma*d_e")

t_max = d_e + d_q

T = 50
pad = int(T * 0.2)
ts = np.linspace(0, t_max, T)
ts = np.concatenate((np.zeros(pad), ts, np.ones(pad)*t_max))

q = (0, d_q)

height = 1

# Create a new figure
fig, ax = plt.subplots(2, 1)

plt.subplots_adjust(right=0.8)

light_blue = '#ADD8E6'
dark_blue = '#00008B'
light_red = '#FF7F7F'
light_green = '#90EE90'
black = '#000000'

t0 = 0
t1 = gamma*d_e
t5 = d_e + d_q

if d_e >= d_q:
    t2 = d_q
    t2_str = 'd_q'
    t3 = d_e
    t3_str = 'd_e'
    t4 = t3 + d_q - t1
else:
    t2 = d_e
    t2_str = 'd_e'
    t3 = d_q
    t3_str = 'd_q'
    t4 = t3 + d_e - t1

print("t1 = {}, t2 = {}, t3 = {}, t4 = {}, t5 = {}".format(t1, t2, t3, t4, t5))

# make sure that the important time states are in ts
ts = np.sort(np.concatenate((np.array([t0, t1+1e-10, t1-1e-10, t2, t3, t4, t4+1e-10, t4-1e-10, t5]), ts)))

# def plot_overlap(t, gamma, offset=(0, 0)):
#     e_t = (q[0] - d_e + t, q[0] - d_e + d_e + t)
#     overlap = min(q[0] + d_q, e_t[0] + d_e) - max(q[0], e_t[0])
#     if overlap > 0:
#         rect_overlap = patches.Rectangle((, y), overlap, height, linewidth=1, edgecolor='none', facecolor=light_blue)
#         ax[1].add_patch(rect_overlap)

def plot_query_and_event_rectangles(t, gamma, rectangle_offset=(0, 0)):
    
    # Initialize rectangles
    rect_q = patches.Rectangle((q[0] + rectangle_offset[0], -height/2 + rectangle_offset[1]), d_q, height, linewidth=1, edgecolor='r', facecolor='none')
    rect_e = patches.Rectangle((0 + rectangle_offset[0], -height/2 + rectangle_offset[1]), d_e, height, linewidth=1, edgecolor='g', facecolor='none')

    rect_overlap = patches.Rectangle((0 + rectangle_offset[0], -height/2 + rectangle_offset[1]), 0, height, linewidth=1, edgecolor='none', facecolor=light_blue)

    e_t = (q[0] - d_e + t, q[0] - d_e + d_e + t)

    rect_q.set_xy((q[0], -height/2))
    rect_e.set_xy((e_t[0], -height/2))
    # add text to the center of rect_e
    ax[1].text(e_t[0] + d_e/2, height, r'$e_{' + r'{:.1f}'.format(t) + r'}$', horizontalalignment='center', verticalalignment='center')
    # add text to the center of rect_q
    ax[1].text(q[0] + d_q/2, -height, r'$q$', horizontalalignment='center', verticalalignment='center')

    ax[1].add_patch(rect_q)

    overlap = min(q[0] + d_q, e_t[0] + d_e) - max(q[0], e_t[0])
    if overlap > 0:
        rect_overlap.set_width(overlap)
        rect_overlap.set_xy((max(q[0], e_t[0]), -height/2))
        # set face color to blue if overlap is greater than gamma*d_e
        if overlap > d_e*gamma:
            rect_overlap.set_facecolor(light_green)
            rect_q.set_facecolor(light_red)
        ax[1].add_patch(rect_overlap)
    
    ax[1].add_patch(rect_e)

def animate(t):
    # moving event
    ax[1].clear()
    ax[1].set_ylim(-5, 5)
    ax[1].set_xlim(-2, 2+d_q)
    ax[1].set_yticks([])
    ax[1].set_xlabel('t')

    plot_query_and_event_rectangles(t, gamma)

    # moving query accuracy plot
    ax[0].clear()
    ax[0].set_ylim(0, 1.1)
    ax[0].set_xlim(-2, 2+d_q)
    
    # get the _ts from ts where _t < t
    _ts = ts[ts < t]
    _qps = []
    for _t in _ts:
        e_t = (q[0] - d_e + _t, q[0] - d_e + d_e + _t)
        metric_score = defs.query_segment_accuracy_score(e_t, q, gamma)
        _qps.append(metric_score)
    ax[0].plot(_ts, _qps)
    
    ax[0].set_ylabel('Query Label Accuracy')
    #ax[0].set_xlabel('t')
    # remove x ticks
    ax[0].set_xticks([])

    # draw time states
    t0 = 0
    t1 = gamma*d_e
    t5 = d_e + d_q
    f_min = (gamma*d_e) / d_q
    f_1 = (gamma*d_e) / d_q
    f_2 = (d_q - gamma*d_e) / d_q


    if d_e >= d_q:
        #ax[0].set_title(r'$d_e \geq d_q$')
        t2 = d_q
        t2_str = 'd_q'
        t3 = d_e
        t3_str = 'd_e'
        t4 = t3 + d_q - t1
        f_max = 1

        f_3 = 0
        
    else:
        #ax[0].set_title(r'$d_e < d_q$')
        t2 = d_e
        t2_str = 'd_e'
        t3 = d_q
        t3_str = 'd_q'
        t4 = t3 + d_e - t1
        f_max = d_e / d_q

        f_3 = (d_q - d_e) / d_q
        

    t_loc = -0.1
    alpha = 0.4

    # draw overlap rectangle
    # overlap = min(q[0] + d_q, q[0] - d_e + d_e + t) - max(q[0], q[0] - d_e + t)
    # ax[1].add_patch(patches.Rectangle((0, 0), overlap, height, linewidth=1, edgecolor='none', facecolor=light_blue))


    if t > t0:
        ax[0].axvline(x=t0, color=black, linestyle='--', label=r'$t_0 = 0$', alpha=alpha)
        #ax[1].axvline(x=t0, color=black, linestyle='--')
        # draw text above the vertical line above the plot
        ax[0].text(t0, t_loc, r'$t_0$', horizontalalignment='center', verticalalignment='center')
    if t > t1:
        ax[0].axvline(x=t1, color=black, linestyle='--', label=r'$t_1 = \gamma \cdot d_e$', alpha=alpha)
        #ax[1].axvline(x=t1, color=black, linestyle='--')
        ax[0].text(t1, t_loc, r'$t_1$', horizontalalignment='center', verticalalignment='center')
        
        #ax[0].axhline(y=f_min, color=black, linestyle='-.', alpha=alpha)
        #ax[0].text(2+d_q+0.05, f_min, r'$f_{\text{min}} = \gamma \cdot d_e / d_q$', horizontalalignment='left', verticalalignment='center')

        ax[0].axhline(y=f_1, color=black, linestyle='-.', alpha=alpha)
        ax[0].text(2+d_q+0.05, f_1, r'$f_1 = \gamma \cdot d_e / d_q$', horizontalalignment='left', verticalalignment='center')
        ax[0].axhline(y=f_2, color=black, linestyle='-.', alpha=alpha)
        ax[0].text(2+d_q+0.05, f_2, r'$f_2 = (d_q - \gamma \cdot d_e) / d_q$', horizontalalignment='left', verticalalignment='center')

    if t > t2:
        ax[0].axvline(x=t2, color=black, linestyle='--', label=r'$t_2 = {}$'.format(t2_str), alpha=alpha)
        if d_e >= d_q:
            f_max_str = r'$f_{\text{max}} = 1$'
            f_3_str = r'$f_3 = 0$'
        else:
            f_max_str = r'$f_{\text{max}} = d_e / d_q$'
            f_3_str = r'$f_3 = (d_q - d_e) / d_q$'

        
        #ax[0].axhline(y=f_max, color=black, linestyle='-.', alpha=alpha)
        #ax[0].text(2+d_q+0.05, f_max, f_max_str, horizontalalignment='left', verticalalignment='center')

        ax[0].axhline(y=f_3, color=black, linestyle='-.', alpha=alpha)
        ax[0].text(2+d_q+0.05, f_3, f_3_str, horizontalalignment='left', verticalalignment='center')

        #ax[1].axvline(x=t2, color=black, linestyle='--')
        ax[0].text(t2, t_loc, r'$t_2$', horizontalalignment='center', verticalalignment='center')
        # color the area below _qps between t1 and t2
        ax[0].fill_between(_ts, _qps, where=[t1 <= t <= t2 for t in _ts], facecolor=light_blue, edgecolor='b', alpha=0.5)
    if t > t3:
        ax[0].axvline(x=t3, color=black, linestyle='--', label=r'$t_3 = {}$'.format(t3_str), alpha=alpha)
        #ax[1].axvline(x=t3, color=black, linestyle='--')
        ax[0].text(t3, t_loc, r'$t_3$', horizontalalignment='center', verticalalignment='center')
        ax[0].fill_between(_ts, _qps, where=[t2 <= t <= t3 for t in _ts], facecolor=dark_blue, edgecolor='b', alpha=0.5)
    if t > t4:
        ax[0].axvline(x=t4, color=black, linestyle='--', alpha=alpha)
        #ax[1].axvline(x=t4, color=black, linestyle='--')
        ax[0].text(t4, t_loc, r'$t_4$', horizontalalignment='center', verticalalignment='center')
        ax[0].fill_between(_ts, _qps, where=[t3 <= t <= t4 for t in _ts], facecolor=light_blue, edgecolor='b', alpha=0.5)
    if t >= t5:
        ax[0].axvline(x=t5, color=black, linestyle='--', alpha=alpha)
        #ax[1].axvline(x=t5, color=black, linestyle='--')
        ax[0].text(t5, t_loc, r'$t_5$', horizontalalignment='center', verticalalignment='center')

    ax[0].legend(loc='upper left')
    if d_e >= d_q:
        title_str = r'$d_e \geq d_q$'
    else:
        title_str = r'$d_e < d_q$'
    ax[0].set_title(r'Accuracy for all overlap occurances ($\gamma = {}$, {})'.format(gamma, title_str))
    #plt.tight_layout()
ani = animation.FuncAnimation(fig, animate, frames=ts, repeat=True)
ani.save('animation/animation.gif', writer='imagemagick', fps=5)

html = ani.to_jshtml()

with open('animation/animation.html', 'w') as f:
    f.write(html)

HTML(html)
# save the HTML animation

