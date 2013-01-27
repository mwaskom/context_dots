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
    frame_contrast=.8,
    frame_sfs=[2, 4.5],

    # Dots
    dot_count=140,
    dot_shape="circle",
    dot_speed=3,  # in degrees/sec
    dot_size=.1,  # in degrees
    dot_life_mean=60,  # in frames
    dot_life_std=20,  # in frames
    dot_colors=["red", "cyan"],
    dot_dirs=[90, 270],
    dot_mot_coh=.3,  # pct dots moving coherently
    dot_col_coh=.3,  # pct dots in target color
    dot_base_hue=0,
    dot_sat=1,
    dot_val=1,

    # Response settings
    quit_keys=["esc", "q"],
    resp_keys=["comma", "period"],
    wait_keys=["space"],

    # Timing
    stim_dur=3,  # seconds

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
    iti=(.5, 2.5),

    log_base="data/%(subject)s_run%(run)d",

    instruct_text="""
        Look at
        all the pretty dots""",
    instruct_size=0.5,

    break_text_size=0.8,
    break_text="""hang tight""",

    trials_bw_breaks=20,

)
punch_behav.update(behav_base)

# Top-level parameters for training/staircase
punch_train = dict(

    # Experimental variables
    n_per_block=6,

    # Feedback
    fb_freq=6,
    fb_dur=1,

)
punch_train.update(punch_base)

# Top-level parameters for behavioral design
behav_design = dict(

    trials_per_run=100,
    early_cue_prob=.5,

)
behav_design.update(behav_base)
