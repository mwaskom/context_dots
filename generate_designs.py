from __future__ import division
import sys
from string import letters
import json
import pandas as pd
import numpy as np
from numpy.random import permutation
from itertools import permutations, product
import tools


def main(arglist):

    mode = arglist.pop(0)
    p = tools.Params("%s_design" % mode)
    p.set_by_cmdline(arglist)
    globals()[mode](p)


def behav(p):

    freq_dist = [.2, .4, .6, .8]
    bal_feature = lambda x: sum(x) == 1
    feat = filter(bal_feature, permutations(freq_dist, 2))
    bal_context = lambda x: sum([i[0] for i in x]) == 1
    full = filter(bal_context, permutations(product(freq_dist, feat), 2))
    idx = np.arange(0, 64, 8) + np.arange(8)
    sched = [full[i] for i in idx]

    for d_i, run in enumerate(sched):

        m_freq, dir_freqs = run[0]
        c_freq, hue_freqs = run[1]

        context = []
        motion = []
        color = []
        early = []

        context_freq = np.array([m_freq, c_freq])
        motion_freq = np.array(dir_freqs)
        color_freq = np.array(hue_freqs)

        for c_i, c_f in enumerate(context_freq):
            context += [c_i] * (c_f * p.trials_per_run)
            ep = p.early_cue_prob
            early += [0] * (((1 - ep) * c_f) * p.trials_per_run)
            early += [1] * ((ep * c_f) * p.trials_per_run)
        for m_i, m_f in enumerate(motion_freq):
            motion += [m_i] * (m_f * p.trials_per_run)
        for h_i, h_f in enumerate(color_freq):
            color += [h_i] * (h_f * p.trials_per_run)

        context_perm = permutation(p.trials_per_run)
        context = np.array(context)[context_perm]
        early = np.array(early)[context_perm]
        motion = permutation(motion)
        color = permutation(color)

        assert sum(context) / p.trials_per_run == context_freq[1]
        assert sum(motion) / p.trials_per_run == motion_freq[1]
        assert sum(color) / p.trials_per_run == color_freq[1]

        context_freq = context_freq[np.array(context)]
        motion_freq = motion_freq[np.array(motion)]
        color_freq = color_freq[np.array(color)]


        run_design = pd.DataFrame(dict(context=context,
                                       motion=motion,
                                       color=color,
                                       context_freq=context_freq,
                                       motion_freq=motion_freq,
                                       color_freq=color_freq,
                                       early=early))

        target_freq = np.zeros(p.trials_per_run)
        for c_i, c_name in enumerate(["motion", "color"]):
            f_freq = run_design["%s_freq" % c_name]
            target_freq[context == c_i] = f_freq
        run_design["target_freq_g_cue"] = target_freq
        run_design["target_freq"] = (run_design["context_freq"] *
                                     run_design["target_freq_g_cue"])

        fname_stem = "design/behav_%s" % letters[d_i]
        run_design.to_csv(fname_stem + ".csv",
                          index_label="trial")

        run_data = dict(context_freq=context_freq,
                        motion_freq=motion_freq,
                        color_freq=color_freq)
        run_data = {k: v.tolist() for k, v in run_data.items()}
        with open(fname_stem + ".json", "w") as fobj:
            json.dump(run_data, fobj)


if __name__ == "__main__":
    main(sys.argv[1:])
