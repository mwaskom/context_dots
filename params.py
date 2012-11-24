punch_base = dict(

    experiment_name="punch",

    # Display setup
    monitor_name='mlw-mbair',
    monitor_units="deg",
    full_screen=True,
    screen_number=1,

    # Fixation
    fix_size=.2,
    fix_color='#FFFFFF',

    # Frame
    frame_size=10,
    frame_width=.5,
    frame_shape=None,
    frame_opacity=.7,
    frame_colors=["red", "blue", "green", "yellow", "magenta", "cyan"],

    # Gratings
    stim_size=7,
    stim_mask="gauss",
    stim_opacity=1,
    stim_contrasts=[.8, .5, .2],
    stim_sfs=[1.5, 2, 2.5],
    stim_oris=[15, 50, 85],
    stim_disk_ratio=8,

    # Dots
    dot_cols=["#FF0000", "#00FF00"],
    dot_dirs=[0, 180],
    dot_mot_coh=.2,
    dot_col_coh=.2,
    dot_ndots=100,
    dot_speed=0.1,
    dot_size=4,

    # Response settings
    resp_keys=["j", "k", "l"],

    )

punch_behav = dict(

    # Experimental variables
    early_cue_prob=.5,

    # Timing
    stim_dur=1,
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
