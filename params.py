from __future__ import division
from copy import deepcopy
base = dict(

    experiment_name="punch",

    # Display setup
    monitor_name="mlw-mbair",
    fmri_monitor_name="cni_47",
    screen_number=0,
    fmri_screen_number=1,
    monitor_units="deg",
    full_screen=True,
    window_color=-.333,

    # Fixation
    fix_size=.2,
    fix_iti_color="black",
    fix_stim_color="white",
    fix_shape="circle",

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
    fmri_resp_keys=["2", "3"],
    wait_keys=["space"],
    finish_keys=["return"],
    trigger_keys=["5", "t"],

    # Where coherence info is saved after training
    coh_file_template="data/%s_coherence.json",

    # General timing (all in seconds)
    tr=2.0,
    orient_dur=.5,
    cue_dur=1.0,
    stim_dur=2.0,

    # Feedback
    fb_freq=10,
    fb_dur=.5,

    # Communication
    instruct_size=0.5,
    instruct_text="""
    Use the comma and period keys to respond
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

)

instruct = deepcopy(base)
instruct.update(dict(

    finish_text="Ready to start! Please tell the experimenter!",

    ))

learn = deepcopy(base)
learn.update(dict(

    n_per_block=8,
    log_base="data/%(subject)s_learn",
    coherence=1,
    iti=(1.5, 3),
    perf_thresh=1,  # mean correct over a block
    blocks_at_thresh=1,  # for each frame
    blocks_bw_break=5,

    ))

staircase = deepcopy(base)
staircase.update(dict(

    n_per_block=6,
    n_blocks=60,
    log_base="data/%(subject)s_staircase",
    iti=(1.5, 3),
    starting_coherence=.4,
    burn_in_blocks=2,
    n_up=1,
    n_down=4,
    step=.05,
    blocks_bw_break=5,
    n_calc_blocks=10,

    ))

practice = deepcopy(base)
practice.update(dict(

    log_base="data/%(subject)s_practice_run%(run)d",
    iti=(1.5, 3),
    equilibrium_trs=2,

    ))
def practice_cmdline(parser):
    parser.add_argument("-trials", type=int, default=200)
    parser.add_argument("-trials_bw_break", type=int, default=30)
    parser.add_argument("-feedback", action="store_true")

scan = deepcopy(base)
scan.update(dict(

    log_base="data/%(subject)s_scan_run%(run)02d",

    # Timing
    equilibrium_trs=6,
    leadout_trs=5,

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
    finish_text="Run completed!",
    instruct_text="""
    Use your index and ring finger to respond
    as soon as you have made each decision!
    """
    ))
