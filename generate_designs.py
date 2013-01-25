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

    # Here's some messy functional combinatoric stuff
    # This creates all possible combinations of the
    # frequencies for both each context and over the contexts.
    # It then selects eight of them in a way that accomplishes
    # two things:
    #
    # 1) balanced entropy over feature freqs conditional on context
    #
    # 2) minimum variance across counts of each target frequency
    #
    # I'm not entirely sure why this works, and I don't know how
    # brittle it is, but it works for the set of parameters I've
    # decided on for now.
    freq_dist = [.2, .4, .6, .8]
    bal_feature = lambda x: sum(x) == 1
    feat = filter(bal_feature, permutations(freq_dist, 2))
    bal_context = lambda x: sum([i[0] for i in x]) == 1
    all_possible = permutations(product(freq_dist, feat), 2)
    full = filter(bal_context, all_possible)
    idx = np.arange(0, 64, 8) + np.arange(8)
    sched = [full[i] for i in idx]

    # Now figure out the structure of each experimental run
    for d_i, run in enumerate(sched):

        # Unpack this run's condition info
        m_freq, dir_freqs = run[0]
        c_freq, hue_freqs = run[1]

        # Set up lists of the schedules we care about
        context = []
        motion = []
        color = []
        early = []

        # Make things arrays so fancy indexing will work
        context_freqs = np.array([m_freq, c_freq])
        motion_freqs = np.array(dir_freqs)
        color_freqs = np.array(hue_freqs)

        # Get references to early cue probabilities
        e_p = p.early_cue_prob
        c_p = 1 - e_p

        # Now generate the unordered lists in a way that
        # balances the relationships we care about
        for e_i, e_f in enumerate([c_p, e_p]):
            early += [e_i] * int((e_f * p.trials_per_run))

            # Balance context with respect to cue type
            for c_i, c_f in enumerate(context_freqs):
                context += [c_i] * (e_f * c_f * p.trials_per_run)

                # Now balance each coherent feature with
                # respect to cue type and target context
                for m_i, m_f in enumerate(motion_freqs):
                    motion += [m_i] * (e_f * c_f * m_f * p.trials_per_run)
                for h_i, h_f in enumerate(color_freqs):
                    color += [h_i] * (e_f * c_f * h_f * p.trials_per_run)

        # Make things arrays
        early = np.array(early)
        context = np.array(context)
        motion = np.array(motion)
        color = np.array(color)

        # Now we're going to shuffle the order of events,
        # again respecting the balance of things we care about
        for is_early in range(2):
            # First shuffle context within each cue type
            shuffle_these = (early == is_early).astype(bool)
            shuffler = np.random.permutation(shuffle_these.sum())
            context[shuffle_these] = context[shuffle_these][shuffler]
            #motion[shuffle_these] = motion[shuffle_these][shuffler]
            #color[shuffle_these] = color[shuffle_these][shuffler]
            for this_context in range(2):
                # Now shuffle the features within each context and cue type
                shuffle_those = shuffle_these & (context == this_context)
                motion[shuffle_those] = permutation(motion[shuffle_those])
                color[shuffle_those] = permutation(color[shuffle_those])

        # Finally shuffle all of the rows
        shuffler = np.random.permutation(p.trials_per_run)
        early = early[shuffler]
        context = context[shuffler]
        motion = motion[shuffler]
        color = color[shuffler]

        # Check some constraints; let's add more of these
        assert context.sum() / p.trials_per_run == context_freqs[1]
        assert motion.sum() / p.trials_per_run == motion_freqs[1]
        assert color.sum() / p.trials_per_run == color_freqs[1]

        # Make a vector of the frequency values we care
        # about for each trial (will make analysis easier)
        context_freq = context_freqs[context]
        motion_freq = motion_freqs[motion]
        color_freq = color_freqs[color]

        # Set up the full design as a DataFrame
        run_design = pd.DataFrame(dict(context=context,
                                       motion=motion,
                                       color=color,
                                       context_freq=context_freq,
                                       motion_freq=motion_freq,
                                       color_freq=color_freq,
                                       early=early))

        # Add in the target frequency, which is the
        # joint probability across context and coherent feature
        target_freq = np.zeros(p.trials_per_run)
        for c_i, c_name in enumerate(["motion", "color"]):
            f_freq = run_design["%s_freq" % c_name]
            target_freq[context == c_i] = f_freq * context_freqs[c_i]
        run_design["target_freq"] = target_freq

        # Set up the outputs
        fname_stem = "design/behav_%s" % letters[d_i]

        # Save the full schedule as a csv
        run_design.to_csv(fname_stem + ".csv",
                          index_label="trial")

        # Also save a json with some metadata
        # (should probably add more of this)
        run_data = dict(context_freqs=(m_freq, c_freq),
                        motion_freqs=dir_freqs,
                        color_freqs=hue_freqs)
        with open(fname_stem + ".json", "w") as fobj:
            json.dump(run_data, fobj)


if __name__ == "__main__":
    main(sys.argv[1:])
