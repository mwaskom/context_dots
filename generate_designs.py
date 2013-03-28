"""Create design csv files for various modes of the punch experiment."""
from __future__ import division
import sys
from string import letters
import pandas as pd
import numpy as np
from itertools import permutations, product
import tools


def main(arglist):

    mode = arglist.pop(0)
    p = tools.Params("%s_design" % mode)
    p.set_by_cmdline(arglist)
    globals()[mode](p)


def scan(p):

    pass


def condition_starter(n_trials, color_pct, frame_per_context, trial_probs):
    """Make an unrandomized dataframe with trials for one condition.

    Here `condition` is taken to mean a particular distribution of
    context frequencies, e.g. [80 motion / 20 color].

    Parameters
    ----------
    n_trials : int
        number of trials for the condition.
    color_pct : int
        percentage of trials that are in color context
    frame_per_context : int
        number of frames (`cue`) assigned to each of the contexts
    trial_probs : dict with keys from {early, later, catch}
        mapping from trial type to percentage of trials of that type

    Returns
    -------
    sched : pandas DataFrame
        unrandomized schedule with trial info

    Raises
    ------
    AssertionError
        many asserts to check for balance of various conditions
        but probably not exhaustive!

    """
    # Set up the empty output lists
    trial_type = []
    context = []
    motion = []
    color = []
    early = []
    stim = []
    cue = []

    # Get the motion frequency in a way that avoids float issues
    motion_pct = 100 - color_pct

    # Fail message
    fail = "Failed to balance design"

    # Check the inputs
    assert isinstance(color_pct, int), "`color_pct` must be an integer"
    assert sum(trial_probs.values()) == 1, "`trial_probs` is ill-formed"

    for type, t_prob in trial_probs.items():
        of_type = t_prob * n_trials
        assert of_type == int(of_type), fail
        of_type = int(of_type)

        # Set up the variables that are stable over this sub-block
        trial_type.extend([type] * of_type)
        early_cue = False if type == "later" else True
        with_stim = False if type == "catch" else True
        early += [early_cue] * of_type
        stim += [with_stim] * of_type

        # Build the sub-components of each trial type
        # First the trial context
        for c_i, c_p in enumerate([motion_pct, color_pct]):
            of_context = (c_p / 100) * of_type
            assert of_context == int(of_context), fail
            of_context = int(of_context)

            context += [c_i] * of_context

            # Balance cue identity within context
            for f_i in range(frame_per_context):
                f_f = 1 / frame_per_context
                of_frame = f_f * of_context
                assert of_frame == int(of_frame), fail
                of_frame = int(of_frame)

                cue += [f_i] * of_frame

                # Now balance the identity of the two features
                if type == "catch":
                    motion += [np.nan] * of_frame
                    color += [np.nan] * of_frame
                else:
                    dirs = np.zeros(of_frame, int)
                    dirs[::2] = 1
                    motion += dirs.tolist()
                    hues = np.zeros(of_frame, int)
                    hues[of_frame / 2:] = 1
                    color += hues.tolist()

    # Get a record to the context frequency for convenience
    context_freq = np.where(np.array(context) == 1,
                            color_pct, motion_pct).astype(float) / 100

    # Construct a DataFrame with these schedules
    sched = pd.DataFrame(dict(trial_type=trial_type,
                              context=context,
                              early=early,
                              stim=stim,
                              cue=cue,
                              motion=motion,
                              color=color,
                              context_freq=context_freq),
                         columns=["trial_type", "context",
                                  "early", "stim", "cue",
                                  "motion", "color",
                                  "context_freq"])

    # Check other constraints
    for c_i, c_pct in enumerate([motion_pct, color_pct]):
        assert (sched.context == c_i).mean() == c_pct / 100, fail
    early_bal = sched.groupby("early").context.mean() * 100
    assert map(int, early_bal.tolist()) == [color_pct, color_pct], fail
    stim_bal = sched.groupby("stim").context.mean() * 100
    assert map(int, stim_bal.tolist()) == [color_pct, color_pct], fail
    assert len(sched.groupby("context").cue.mean().unique()) == 1, fail

    return sched


def behav_old(p):

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
        cue = []

        # Make things arrays so fancy indexing will work
        context_freqs = np.array([m_freq, c_freq])
        motion_freqs = np.array(dir_freqs)
        color_freqs = np.array(hue_freqs)

        # Get references to early cue probabilities
        e_p = p.early_cue_prob
        c_p = 1 - e_p

        # Get info about the different cues
        f_f = 1. / p.frame_per_context

        # Now generate the unordered lists in a way that
        # balances the relationships we care about
        for e_i, e_f in enumerate([c_p, e_p]):
            early += [e_i] * int((e_f * p.trials_per_run))

            # Balance cue identity within cue timing
            for f_i in range(p.frame_per_context):
                f_f = 1 / p.frame_per_context

                cue += [f_i] * int((e_f * f_f * p.trials_per_run))

                # Balance context with respect to cue type
                for c_i, c_f in enumerate(context_freqs):
                    nt = e_f * f_f * c_f * p.trials_per_run
                    context += [c_i] * nt

                    for m_i, m_f in enumerate(motion_freqs):
                        motion += [m_i] * (nt * m_f)

                    color_temp = []
                    for h_i, h_f in enumerate(color_freqs):
                        color_temp += [h_i] * (nt * h_f)
                    color += reversed(color_temp)

        # Make things arrays
        early = np.array(early, int)
        context = np.array(context, int)
        motion = np.array(motion, int)
        color = np.array(color, int)
        cue = np.array(cue, int)

        # Check some constraints; let's add more of these
        assert context.sum() / p.trials_per_run == context_freqs[1]
        assert motion.sum() / p.trials_per_run == motion_freqs[1]
        assert color.sum() / p.trials_per_run == color_freqs[1]
        assert early.mean() == p.early_cue_prob

        # Make a vector of the frequency values we care
        # about for each trial (will make analysis easier)
        context_freq = context_freqs[context]
        motion_freq = motion_freqs[motion]
        color_freq = color_freqs[color]

        # Set up the full design as a DataFrame
        run_design = pd.DataFrame(dict(context=context,
                                       early=early,
                                       cue=cue,
                                       motion=motion,
                                       color=color,
                                       context_freq=context_freq,
                                       motion_freq=motion_freq,
                                       color_freq=color_freq),
                                  columns=["context", "early", "cue",
                                           "motion", "color",
                                           "context_freq",
                                           "motion_freq", "color_freq"])

        # Add in additional info that is dependent on
        # the schedule we just built
        target_freq = np.zeros(p.trials_per_run)
        target = np.zeros(len(run_design))
        for c_i, c_name in enumerate(["motion", "color"]):
            f_freq = run_design["%s_freq" % c_name]
            mask = context == c_i
            target_freq[mask] = f_freq * context_freqs[c_i]
            target[mask] = run_design[c_name][mask]
        run_design["target_freq"] = target_freq
        butt1_freq = target.mean()
        response_freq = np.where(target, 1 - butt1_freq, butt1_freq)
        run_design["response_freq"] = response_freq
        correct_response = np.zeros(p.trials_per_run)
        correct_response[context == 0] = motion[context == 0]
        correct_response[context == 1] = color[context == 1]
        run_design["correct_response"] = correct_response

        # Add in a column about congruency for later analysis
        run_design["congruent"] = run_design["motion"] == run_design["color"]

        # Set up the outputs
        fname = p.design_template % letters[d_i]

        # Save the full schedule as a csv
        run_design.to_csv(fname,
                          index_label="trial")


if __name__ == "__main__":
    main(sys.argv[1:])
