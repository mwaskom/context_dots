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
    frame_width=.5,
    frame_shape=None,
    frame_opacity=1,
    frame_colors=["gray", "white"],

    # Dots
    dot_field_size=8,  # in degrees
    dot_shape="circle",
    dot_colors=[(1, -1, -1), (-1, 1, -1), (-1, -1, 1), (1, 1, -1)],
    dot_color_names=["red", "green", "blue", "yellow"],
    dot_dirs=[45, 135, 225, 315],
    dot_life_mean=20,  # in frames
    dot_life_std=5,  # in frames
    dot_mot_coh=.45,  # pct dots moving coherently
    dot_col_coh=.8,  # pct additional dots in target color
    dot_number=100,
    dot_speed=3,  # in degrees/sec
    dot_size=.1,  # in degrees

    # Response settings
    resp_keys=["j", "k", "l"],

    )

punch_behav = dict(

    # Experimental variables
    early_cue_prob=.5,
    stim_dur=1,

    # Timing
    stim_flips=120,  # number of frames
    iti=(.5, 2.5),
    cue_dur=(.5, 2.5),

)
punch_behav.update(punch_base)

punch_train = dict(

    # Feedback halo
    fb_size=9,
    fb_colors=["#FF4215", "#00FF75"],
    fb_dur=.5,
    fb_opacity=.5,
    fb_mask="gauss",

    # Experimental variables
    early_cue_prob=0,
    n_per_block=4,

    # Timing
    stim_dur=1,
    iti=(.5, 2.5),
    cue_dur=(.5, 2.5),

)
punch_train.update(punch_base)


def add_cmdline_params(parser):

    parser.add_argument("-train", action="store_true")
    return parser
