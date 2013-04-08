"""Create design csv files for various modes of the punch experiment."""
from __future__ import division
import sys
import pandas as pd
import numpy as np
from scipy import stats
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
    conditions = np.sort(np.unique(p.color_freqs))
    sched_dict = {}
    for freq in conditions:
        sched = condition_schedule(p, freq)
        sched_pair = (sched[:short_dur], sched[short_dur:])
        sched_dict[freq] = sched_pair

    # Order the blocks into a full schedule for the experiment
    schedule = []
    block_info = zip(p.color_freqs, p.block_duration)
    for i, (freq, dur) in enumerate(block_info):
        block_sched = sched_dict[freq][dur]
        block_sched["block"] = i
        schedule.append(block_sched)
    schedule = pd.concat(schedule)
    schedule.index = pd.Index(range(len(schedule)), name="trial")

    # Add in run information
    run = np.repeat(np.arange(p.n_runs), p.trials_per_run) + 1
    schedule["run"] = run
    run_trial = np.tile(np.arange(p.trials_per_run), p.n_runs)
    schedule["run_trial"] = run_trial

    # Sort out the null event timing
    schedule["iti"] = np.nan
    schedule["cue_time"] = np.nan
    for run in schedule.run.unique():
        run_index = schedule.run == run
        sched_i = schedule[run_index]
        secs_per_run = p.trs_per_run * p.tr
        null_time = secs_per_run - sched_i.trial_dur.sum()
        iti = trial_timing(p, null_time)
        schedule.iti[run_index] = pd.Series(iti, index=sched_i.index)
        total_trs = (iti.sum() + sched_i.trial_dur.sum()) / p.tr
        assert total_trs == p.trs_per_run, "Failed to distribute timing."

        # Schedule the start of each trial (onset of the cue)
        trial_dur = sched_i.trial_dur + iti
        cue_time = trial_dur.cumsum() - sched_i.trial_dur
        schedule.cue_time[run_index] = cue_time

    # Get a record of the trial index from a condition perspective
    condition_trial = np.empty(len(schedule), int)
    for condition in schedule.color_freq.unique():
        trials = schedule.color_freq == condition
        condition_trial[trials] = range(p.trials_per_condition)
    schedule["condition_trial"] = condition_trial

    # Record switch information (this is annoiyng to do)
    # For the feature switches, we pretend like the catch trials don't exist
    context_switch = np.zeros(len(schedule))
    context_switch[1:] = schedule.context[1:] != schedule.context[:-1]
    context_switch[schedule.run_trial == 0] = False

    frame_switch = np.zeros(len(schedule))
    frame_switch[1:] = ((schedule.context[1:] != schedule.context[:-1]) |
                        (schedule.cue[1:] != schedule.cue[:-1]))
    frame_switch[schedule.run_trial == 0] = False

    stim = np.array(schedule.stim)
    motion = schedule.motion[stim]
    motion_switch_ = np.zeros(stim.sum())
    motion_switch_[1:] = motion[1:] != motion[:-1]
    motion_switch_[schedule.run_trial[stim] == 0] = False
    motion_switch = np.zeros(len(schedule), bool)
    motion_switch[stim] = motion_switch_.astype(bool)

    color = schedule.color[stim]
    color_switch_ = np.zeros(stim.sum())
    color_switch_[1:] = color[1:] != color[:-1]
    color_switch_[schedule.run_trial[stim] == 0] = False
    color_switch = np.zeros(len(schedule), bool)
    color_switch[stim] = color_switch_.astype(bool)

    switches = pd.DataFrame(dict(context_switch=context_switch,
                                 motion_switch=motion_switch,
                                 color_switch=color_switch,
                                 frame_switch=frame_switch),
                            index=schedule.index)
    schedule = schedule.join(switches, how="outer")

    # Assertion tests for schedule construction
    fail = "Something's gone wrong"
    assert not schedule.early[schedule.trial_type == "later"].any(), fail
    assert schedule.stim[schedule.trial_type == "later"].all(), fail
    assert schedule.early[schedule.trial_type == "early"].all(), fail
    assert schedule.stim[schedule.trial_type == "early"].all(), fail
    assert schedule.early[schedule.trial_type == "catch"].all(), fail
    assert not schedule.stim[schedule.trial_type == "catch"].any(), fail

    schedule.to_csv(p.design_file, index=False)


def trial_timing(p, null_time):
    """Get a vector of ITI durations to fill in the null time of a run."""
    geom_param = p.iti_geom_param
    max_iti = p.max_iti
    n_trials = p.trials_per_run
    while True:
        candidates = p.random_state.geometric(geom_param, (100, n_trials))
        candidates += p.iti_geom_loc
        too_long = candidates > max_iti
        while too_long.any():
            candidates[too_long] = p.random_state.geometric(geom_param,
                                                            too_long.sum())
            too_long = candidates > max_iti
        right_length = candidates.sum(axis=1) == null_time
        if not any(right_length):
            continue
        else:
            assert np.all(candidates <= max_iti), "Something's gone wrong"
            return candidates[right_length][0]


def condition_schedule(p, color_freq):
    """Generate the ordered schedule of trial info subject to constraints.

    Here `condition` is taken to mean a particular distribution of
    context frequencies, e.g. [80 motion / 20 color].

    Parameters
    ----------
    p : Params object
        record of experimental parameters
    color_freq : freq
        frequency of trials that are in color context

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
    assert trial_probs.sum() == 1, "`trial_probs` is ill-formed"

    # Construct the base schedule
    sched = condition_starter(n_trials, color_freq,
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

    # Establish the cue duration, either as uniform params or a constant
    try:
        a, b = p.cue_dur
        cue_dur = p.random_state.uniform(a, b, n_trials)
    except TypeError:
        cue_dur = p.cue_dur
    sched["cue_dur"] = cue_dur
    sched.cue_dur[sched.trial_type == "later"] = np.nan

    # Get a record of how long each trial will take
    trial_durations = dict(later=p.stim_dur,
                           early=p.cue_dur + p.stim_dur,
                           catch=p.cue_dur)
    sched["trial_dur"] = sched.trial_type.map(trial_durations)

    # Boolean masks corresponding to contexts (useful for code below)
    motion_trials = sched.context == 0
    color_trials = sched.context == 1

    # Get a record of color task frequency as a condition identifier
    sched["color_freq"] = sched["context_freq"]
    sched.color_freq[color_trials] = 1 - sched.context_freq[color_trials]

    # Get a record of context entropy
    sched["context_entropy"] = stats.entropy([sched.context_freq,
                                              1 - sched.context_freq])

    # Get a record of congruency
    sched["congruent"] = sched.motion == sched.color

    # Get a record of the correct response
    sched["target"] = np.nan
    sched.target[motion_trials] = sched.motion[motion_trials]
    sched.target[color_trials] = sched.color[color_trials]

    return sched


def condition_starter(n_trials, color_freq, frame_per_context, trial_probs):
    """Make an unrandomized dataframe with trials for one condition.

    Parameters
    ----------
    n_trials : int
        number of trials for the condition.
    color_freq : float
        frequency of trials that are in color context
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

    # Get the motion frequency
    motion_freq = 1 - color_freq

    # Fail message
    fail = "Something's gone wrong"

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
        for c_i, c_p in enumerate([motion_freq, color_freq]):
            of_context = c_p * of_type
            of_context = np.round(of_context)

            context += [c_i] * of_context

            # Balance cue identity within context
            for f_i in range(frame_per_context):
                f_f = 1 / frame_per_context
                of_frame = f_f * of_context
                of_frame = np.round(of_frame)

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
                            color_freq, motion_freq).astype(float)

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
    for c_i, c_freq in enumerate([motion_freq, color_freq]):
        assert np.allclose((sched.context == c_i).mean(), c_freq), fail
    early_bal = sched.groupby("early").context.mean()
    assert np.allclose(early_bal, [color_freq, color_freq]), fail
    stim_bal = sched.groupby("stim").context.mean()
    assert np.allclose(stim_bal, [color_freq, color_freq]), fail
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

    # Use integers for better subtraction performance
    color_freq = np.array(sched.context_freq[sched.context == 1])[0]
    context_freqs = np.array([1 - color_freq, color_freq])
    desired = 1 - np.array(context_freqs)

    # Shuffle until we get a good distribution of switches
    switch = pd.Series(np.zeros(len(sched), bool))
    index = sched.index
    while True:
        s_i = sched.reindex(random_state.permutation(index))
        context = np.array(s_i.context)
        switch[1:] = context[1:] != context[:-1]
        actual = switch.groupby(context).mean()
        if np.allclose(actual, desired, atol=tol):
            break

    return s_i


if __name__ == "__main__":
    main(sys.argv[1:])
