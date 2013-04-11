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
    dot_shape="square",
    dot_subframes=3,
    dot_speed=5,  # in degrees/sec
    dot_size=.07,  # in degrees
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
    trial_probs=dict(later=1/3,
                     early=1/3,
                     catch=1/3),
    iti_geom_param=1/3,
    iti_geom_loc=1,
    max_iti=10,  # in seconds
    switch_tol=0.01,
    trial_trans_tol=0.1,
    design_file="design/scan_design.csv",
)

learn = dict(

    n_per_block=2,
    monitor_name="mlw-mbair",
    log_base="data/%(subject)s_learn",
    coherence=1,
    iti=(1.5, 3),
    perf_thresh=1,
    blocks_at_thresh=1,
    blocks_bw_break=5,

    # Feedback
    fb_freq=10,
    fb_dur=1,

    )
learn.update(base)

scan = dict(

    tr=2.0,
    cue_dur=1.0,
    monitor_name="mlw-mbair",
    log_base="data/%(subject)s_scan_run%(run)d",

    )
scan.update(base)

scan_design = scan
