from __future__ import division
# Base for everything that's
# going to support data collection
base = dict(

    experiment_name="punch",

    # Display setup
    monitor_units="deg",
    full_screen=True,
    screen_number=1,
    window_color=-1./3,

    # Fixation
    fix_size=.2,
    fix_iti_color="black",
    fix_stim_color="white",
    fix_shape="circle",
    fix_orient_dur=.5,

    # Frame
    field_size=7,
    frame_width=.5,
    frame_tex="sin",
    frame_per_context=2,
    frame_contrasts=[.65, .65, 1, 1],
    frame_sfs=[2, 4.5, 2.5, 6],
    frame_phases=[.5, 0, 0, 0],
    frame_patterns=["short", "long", "plaid", "plaid"],

    # Dots
    dot_count=50,
    dot_shape="circle",
    dot_speed=5,  # in degrees/sec
    dot_size=.075,  # in degrees
    dot_dirs=[90, 270],
    dot_dir_names=["up", "down"],
    dot_colors=[(0.93226, 0.53991, 0.26735),
                (0., 0.74055, 0.22775)],
    dot_color_names=["red", "green"],

    # Response settings
    quit_keys=["escape", "q"],
    resp_keys=["comma", "period"],
    wait_keys=["space"],

    # Where coherence info is saved after training
    coh_file_template="data/%s_coherence.json",

    # Timing
    stim_dur=2,     # seconds

    # Communication
    instruct_size=0.5,
    instruct_text="""
     Use the < and > keys to respond
    as soon as you make your decision

               Press space to begin
          """,

    break_text_size=0.5,
    break_text="""
            Take a quick break, if you'd like!
            Press space to start the next block
            """,
    finish_text="""
                Run Finished!

        Please tell the experimenter!
        """,

    # Design
    n_runs=12,
    color_freqs=[.5, 1/3, .8, 2/3, .2, 2/3, .5, .8, .2, 1/3],
    block_duration=[1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    block_division=(1 / 3, 2 / 3),
    trials_per_condition=180,
    trials_per_run=75,
    trs_per_run=225,
    trial_probs=dict(later=1 / 3,
                     early=1 / 3,
                     catch=1 / 3),
    trial_durations=dict(later=1, early=2, catch=1),
    iti_geom_param=0.5,
    max_iti=5,  # in TRs
    switch_tol=0.01,
    trial_trans_tol=0.1,
    design_file="design/scan_design.csv",
)

# Base for behavioral design and collection
behav_base = dict(

    design_template="design/behav_%s.csv",
    n_designs=8,

)
behav_base.update(base)

# Top-level paramters for behavioral experiment
behav = dict(

    monitor_name='mlw-mbair',
    cue_dur=(.85, 1),

    log_base="data/%(subject)s_behav_run%(run)d",

    isi=(.5, 2.5),  # uniform params (s)
    trials_bw_breaks=20,

)
behav.update(behav_base)

# Top-level parameters for training/staircase
train = dict(

    monitor_name='mlw-mbair',
    log_base="data/%(subject)s_training",

    # Experimental variables
    n_per_block=4,
    full_coh_thresh=1,
    at_thresh_blocks=2,
    settle_slope=.1,
    settle_thresh=.8,
    motion_coh_target=.2,
    color_coh_floor=.05,
    reversal_steps=[.06, .04, .02],
    blocks_bw_break=4,
    isi=(.5, 2.5),  # uniform params (s)

    # Feedback
    fb_freq=10,
    fb_dur=1,

)
train.update(base)

scan = dict(

    tr=2.0,
    cue_dur=(.8, 1),  # TODO
    monitor_name="mlw-mbair",
    log_base="data/%(subject)s_scan_run%(run)d",

    )
scan.update(base)

demo = train
scan_design = base
