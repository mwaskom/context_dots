punch_base = dict(

    experiment_name="punch",

    # Display setup
    monitor_name='mlw-mbair',
    monitor_units="deg",
    full_screen=True,
    screen_number=1,
    window_color="black",

    # Fixation
    fix_size=.2,
    fix_color='#FFFFFF',

    # Frame
    field_size=8,
    frame_width=.5,
    frame_tex="sin",
    frame_contrast=.8,
    frame_sfs=[2, 4.5],

    # Dots
    dot_shape="circle",
    dot_color_names=["red", "green", "blue"],
    dot_dirs=[30, 150, 270],
    dot_life_mean=60,  # in frames
    dot_life_std=20,  # in frames
    dot_mot_coh=.2,  # pct dots moving coherently
    dot_col_coh=.2,  # pct  dots in target color
    dot_sat=.75,
    dot_val=1,
    dot_number=100,
    dot_speed=3,  # in degrees/sec
    dot_size=.1,  # in degrees

    # Response settings
    quit_keys=["esc", "q"],
    resp_keys=["j", "k", "l"],

    # Timing
    stim_flips=120,  # number of frames
    iti=(.5, 2.5),

    )

punch_behav = dict(

    # Experimental variables
    early_cue_prob=.5,

    # Timing
    cue_dur=(.5, 2.5),

)
punch_behav.update(punch_base)

punch_train = dict(

    # Experimental variables
    early_cue_prob=0,
    n_per_block=6,

    # Feedback
    fb_freq=6,
    fb_dur=1,

)
punch_train.update(punch_base)


def add_cmdline_params(parser):

    parser.add_argument("-train", action="store_true")
    return parser
