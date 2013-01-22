from __future__ import division
import sys
import os.path as op
from string import letters
import json
import pandas as pd
import numpy as np
from numpy.random import randint, permutation
import tools


def main(arglist):

    mode = arglist.pop(0)
    p = tools.Params("%s_design" % mode)
    p.set_by_cmdline(arglist)
    globals()[mode](p)


def behav(p):

    for type in ["context", "feature"]:
        assert len(np.unique(map(len, getattr(p, "%s_freqs" % type)))) == 1

    n_c_freqs = len(p.context_freqs)
    n_f_freqs = len(p.feature_freqs)

    for d_i in xrange(p.total_designs):

        context = []
        motion = []
        color = []
        early = []

        context_freq = permutation(p.context_freqs[randint(n_c_freqs)])
        motion_freq = permutation(p.feature_freqs[randint(n_f_freqs)])
        color_freq = permutation(p.feature_freqs[randint(n_f_freqs)])

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

        run_design.to_csv("design/behav_%02d.csv" % d_i,
                          index_label="trial")

        run_data = dict(context_freq=context_freq,
                        motion_freq=motion_freq,
                        color_freq=color_freq)
        run_data = {k: v.tolist() for k, v in run_data.items()}
        with open("design/behav_%02d.json" % d_i, "w") as fobj:
            json.dump(run_data, fobj)


if __name__ == "__main__":
    main(sys.argv[1:])
