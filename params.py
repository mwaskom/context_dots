# Base for everything that's
# going to support data collection
punch_base = dict(

    experiment_name="punch",

    # Display setup
    monitor_units="deg",
    full_screen=True,
    screen_number=1,
    window_color=0,

    # Fixation
    fix_size=.2,
    fix_color="white",
    fix_shape="circle",

    # Frame
    field_size=8,
    frame_width=.5,
    frame_tex="sin",
    frame_per_context=2,
    frame_contrasts=[.65, .65, 1, 1],
    frame_sfs=[2, 4.5, 2.5, 8],
    frame_phases=[.5, 0, 0, .5],
    frame_patterns=["short", "long", "plaid", "plaid"],

    # Dots
    dot_count=128,
    dot_shape="circle",
    dot_speed=3,  # in degrees/sec
    dot_size=.1,  # in degrees
    dot_life_mean=60,  # in frames
    dot_life_std=20,  # in frames
    dot_mot_coh=.3,  # pct dots moving coherently
    dot_col_coh=.3,  # pct dots in target color
    dot_dirs=[90, 270],
    dot_colors=["yellow", "cyan"],
    dot_hues=[1. / 6, .5],  # hue coordinates in hsv space
    dot_sat=1,
    dot_val=1,

    # Response settings
    quit_keys=["esc", "q"],
    resp_keys=["comma", "period"],
    wait_keys=["space"],

    # Timing
    stim_dur=2,     # seconds

    # Communication
    instruct_size=0.5,
    instruct_text="""
    Use the < and > keys to respond
      while dots are on the screen

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

# Base for behavioral design and collection
behav_base = dict(

    design_template="design/behav_%s.csv",
    n_designs=8,

)
behav_base.update(punch_base)

# Top-level paramters for behavioral experiment
punch_behav = dict(

    monitor_name='mlw-mbair',
    cue_dur=(1, 1.25),

    log_base="data/%(subject)s_behav_run%(run)d",

    isi=(.5, 2.5),  # uniform params (s)
    trials_bw_breaks=20,

)
punch_behav.update(behav_base)

# Top-level parameters for training/staircase
punch_train = dict(

    monitor_name='mlw-mbair',
    log_base="data/%(subject)s_training",

    # Experimental variables
    n_per_block=8,
    full_coh_thresh=1,
    at_thresh_blocks=2,
    settle_slope=.1,
    motion_coh_target=.3,
    color_coh_step=.05,
    color_coh_reversals=3,
    blocks_bw_break=4,
    isi=(.5, 2.5),  # uniform params (s)

    # Feedback
    fb_freq=10,
    fb_dur=1,

)
punch_train.update(punch_base)

# Top-level parameters for behavioral design
behav_design = dict(

    trials_per_run=100,
    early_cue_prob=.5,

)
behav_design.update(behav_base)

punch_demo = punch_train
