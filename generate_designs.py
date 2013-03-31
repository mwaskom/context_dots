"""Create design csv files for various modes of the punch experiment."""
from __future__ import division
import sys
import pandas as pd
import numpy as np
import tools
import moss


def main(arglist):

    mode = arglist.pop(0)
    p = tools.Params("%s_design" % mode)
    p.set_by_cmdline(arglist)
    globals()[mode](p)


def scan(p):
    """Generate design files for the scanning component of the project."""
    # Check the inputs
    assert sum(p.block_division) == 1, "`block_division` is poorly formed"

    # Use a consistent random state
    seed = sum(map(ord, list("punch")))
    p.random_state = np.random.RandomState(seed)

    # Build a dictionary of condition schedules, separated
    # into short/long condition blocks
    short_dur = p.trials_per_condition * p.block_division[0]
    conditions = np.sort(np.unique(p.color_pcts))
    sched_dict = {}
    for pct in conditions:
        sched = condition_schedule(p, pct)
        sched_pair = (sched[:short_dur], sched[short_dur:])
        sched_dict[pct] = sched_pair

    # Order the blocks into a full schedule for the experiment
    full_schedule = []
    block_info = zip(p.color_pcts, p.block_duration)
    for i, (pct, dur) in enumerate(block_info):
        block_sched = sched_dict[pct][dur]
        block_sched["block"] = i
        full_schedule.append(block_sched)
    full_schedule = pd.concat(full_schedule)

    # Now split the full schedule into runs
    all_trials = np.arange(len(full_schedule))
    full_schedule.index = all_trials
    run_indices = np.split(all_trials, p.n_runs)

    run_schedules = []
    for run_i in range(p.n_runs):
        run_sched = full_schedule.ix[run_indices[run_i]]
        run_trials = np.arange(len(run_sched))
        run_sched.index = run_trials
        run_sched["run"] = run_i + 1
        run_schedules.append(run_sched)

    return run_schedules


def condition_schedule(p, color_pct):
    """Generate the ordered schedule of trial info subject to constraints.

    Here `condition` is taken to mean a particular distribution of
    context frequencies, e.g. [80 motion / 20 color].

    Parameters
    ----------
    p : Params object
        record of experimental parameters
    color_pct : int
        percentage of trials that are in color context

    Returns
    -------
    sched : pandas DataFrame
        schedule of trial information

    """
    # Unpack the parameters
    n_trials = p.trials_per_condition
    frame_per_context = p.frame_per_context
    trial_probs = pd.Series(p.trial_probs)

    # Check the inputs
    assert isinstance(color_pct, int), "`color_pct` must be an integer"
    assert trial_probs.sum() == 1, "`trial_probs` is ill-formed"

    # Construct the base schedule
    sched = condition_starter(n_trials, color_pct,
                              frame_per_context, trial_probs)

    # Get the ideal transition matrix for trial types
    trial_types = trial_probs.index
    ideal_trans = pd.DataFrame(columns=trial_types, index=trial_types)
    for key in trial_types:
        ideal_trans[key].update(trial_probs)
    ideal_trans = np.array(ideal_trans)

    # Now work out the order of trials
    while True:
        sched = balance_switches(sched, p.switch_tol, p.random_state)
        actual = np.array(moss.transition_probabilities(sched.trial_type))
        cost = np.abs(ideal_trans - actual).sum()
        if cost < p.trial_trans_tol:
            break

    return sched


def condition_starter(n_trials, color_pct, frame_per_context, trial_probs):
    """Make an unrandomized dataframe with trials for one condition.

    Parameters
    ----------
    n_trials : int
        number of trials for the condition.
    color_pct : int
        percentage of trials that are in color context
    frame_per_context : int
        number of frames (`cue`) assigned to each of the contexts
    trial_probs : Series with keys from {early, later, catch}
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

    for type, t_prob in trial_probs.iteritems():
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


def balance_switches(sched, tol=.01, random_state=None):
    """Permute trials to balance context switches.

    In the long run, the average switch trial frequency should be
    1 - context frequency. This permutes a schedule DataFrame until
    it is close to that set of frequencies.

    Parameters
    ----------
    sched : pandas DataFrame
        schedule from condition_starter
    tol : float
        tolerance threshold for actual vs. ideal switch freq array
    random_state : numpy RandomState
        allows for consistent randomization over executions

    Returns
    -------
    permuted pandas DataFrame

    """
    if random_state is None:
        random_state = np.random.RandomState()

    color_pct = int(sched.context_freq[sched.context == 1].unique()[0] * 100)
    context_freqs = np.array([100 - color_pct, color_pct], float) / 100
    desired = 1 - context_freqs

    switch = pd.Series(np.ones(len(sched), bool))
    index = sched.index
    while True:
        s_i = sched.reindex(random_state.permutation(index))
        context = np.array(s_i.context)
        switch[1:] = context[1:] != context[:-1]
        actual = switch.groupby(context).mean()
        if np.allclose(actual, desired, atol=tol):
            break

    s_i.index = np.arange(len(s_i))
    return s_i


if __name__ == "__main__":
    main(sys.argv[1:])
