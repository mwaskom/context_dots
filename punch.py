"""Main script for the parametric uncertainty project.

See project README for more information about this experiment and
the way it is controlled by this program.

"""
from __future__ import division
import sys
import json
import itertools
from string import letters
from textwrap import dedent
import pandas as pd
import numpy as np
from numpy.random import randint, uniform
from scipy import stats
from psychopy import visual, core, event
import cregg


def main(arglist):

    # Get the experiment parameters
    mode = arglist.pop(0)
    p = cregg.Params(mode)
    p.set_by_cmdline(arglist)

    # Assign the frame identities randomly over subjects
    state = cregg.subject_specific_state(p.subject)
    frame_ids = state.permutation(list(letters[:2 * p.frame_per_context]))
    p.frame_ids = frame_ids.reshape(2, -1).tolist()

    # Open up the stimulus window
    win = cregg.launch_window(p)

    # Set up the stimulus objects
    fix = visual.GratingStim(win, tex=None,
                             mask=p.fix_shape, interpolate=True,
                             color=p.fix_iti_color, size=p.fix_size)

    instruct_text = dedent(p.instruct_text)
    instruct = cregg.WaitText(win, instruct_text,
                              height=p.instruct_size,
                              advance_keys=p.wait_keys,
                              quit_keys=p.quit_keys)

    stims = dict(frame=Frame(win, p, fix),
                 dots=Dots(win, p),
                 fix=fix,
                 instruct=instruct,
                 )

    if hasattr(p, "break_text"):
        break_text = dedent(p.break_text)
        take_break = cregg.WaitText(win, break_text,
                                    height=p.break_text_size,
                                    advance_keys=p.wait_keys,
                                    quit_keys=p.quit_keys)
        stims["break"] = take_break

    if hasattr(p, "finish_text"):
        finish_text = dedent(p.finish_text)
        finish_run = cregg.WaitText(win, finish_text,
                                    height=p.break_text_size,
                                    advance_keys=p.finish_keys,
                                    quit_keys=p.quit_keys)
        stims["finish"] = finish_run

    # Execute the experiment function
    globals()[mode](p, win, stims)


# Experiment Functions
# ====================


def scan(p, win, stims):
    """Neuroimaging experiment."""

    # Get the design
    design = pd.read_csv(p.design_file)
    design = design[design.run == p.run].set_index("run_trial")

    # Draw the instructions
    stims["instruct"].draw()

    # Set up the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p)

    # Find and set this subjects' coherence values
    subject_coherence(p, stims)

    # Set up the log file
    d_cols = list(design.columns)
    log_cols = d_cols + ["frame_id",
                         "cue_onset", "stim_onset",
                         "response", "rt", "correct",
                         "motion_signal", "color_signal",
                         "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Execute the experiment
    with cregg.PresentationLoop(win, p, log=log,
                                fix=stims["fix"], exit_func=scan_exit):
        stim_event.clock.reset()
        for t, t_info in design.iterrows():

            # ITI fixation
            stims["fix"].draw()
            win.flip()
            cregg.wait_check_quit(t_info["iti"] - p.orient_dur)

            # Stimulus Event
            stim_info = t_info[["cue_time", "context", "cue",
                                "motion", "color",
                                "target", "early",
                                "cue_dur", "stim"]]
            res = stim_event(**stim_info)
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        stims["finish"].draw()


def scan_exit(log):
    """Gets executed at the end of a behavioral run."""
    df = pd.read_csv(log.fname)
    if not len(df):
        return
    print "Overall Accuracy: %.2f" % df.correct.dropna().mean()
    print df.groupby("context").correct.dropna().mean()
    if df.rt.notnull().any():
        print "Average RT: %.2f" % df.rt.dropna().mean()
        print df.groupby("context").rt.dropna().mean()
    diff = (df.cue_time - df.cue_onset).abs().mean()
    if  diff > .025:
        print "Detected issues with cue timing (diff = %.3f)" % diff


def instruct(p, win, stims):
    """Participant-paced instructions with live demos of stimuli."""

    with cregg.PresentationLoop(win, p):

        stim_event = EventEngine(win, stims, p)
        dots = stims["dots"]
        frame = stims["frame"]

        main_text = visual.TextStim(win, height=.5)
        next_text = visual.TextStim(win, "(press space to continue)",
                                    height=.4,
                                    pos=(0, -5))

        def slide(message, with_next_text=True):

            main_text.setText(dedent(message))
            main_text.draw()
            if with_next_text:
                next_text.draw()
            win.flip()
            cregg.wait_and_listen("space", .25)

        slide("""
              Welcome to the experiment - thank you for participating!

              Please read the following instructions carefully and ask
              the experimenter if any part is confusing.
              """)

        slide("""
              In this experiment, you'll be making decisions about
              the information you see in simple, noisy stimuli.

              Each stimulus will consist of a field of dots.

              Some of the dots will be moving either up or down,
              while the rest will flicker randomly.

              Some of the dots will be green, and the rest will be red.

              Press space to see an example stimulus.
              """, False)

        dots.motion_coherence = .35
        dots.color_coherence = .35
        at_time = stim_event.clock.getTime() + 1
        stim_event(at_time, 0, 0, 0, 0, 0)

        slide("""
              On each trial, you'll be making one of two decisions:

              - Are the dots moving up or are they moving down?

              - Are there more red dots or more green dots?

              """)

        slide("""
              The pattern in the frame around the stimulus is what signals
              whether you should make a motion decision or a color decision
              on each trial.

              Next, you'll see each of the four frames. Two of them will
              mean you should make a decision about dot motion, while the other
              two will mean you should make a decision about the dot colors.

              Don't worry too much about memorizing the frames now; you'll
              receive extensive practice before the main experiment begins.

              """)

        for i, context in enumerate(["motion", "color"]):
            for j in range(p.frame_per_context):
                frame.make_active(p.frame_ids[i][j])
                frame.draw()
                slide("\n\n" + context)

        button_ass = dict(motion_left=p.dot_dir_names[0],
                          motion_right=p.dot_dir_names[1],
                          color_left=p.dot_color_names[0],
                          color_right=p.dot_color_names[1])

        slide("""
              For all parts of the experiment, use the index and middle
              fingers on your right hand to make your responses.

              Inside the scanner, you'll be using a button box with just
              a few buttons. For the parts that take place on a computer,
              you should make your responses using the %s and %s buttons.
              """ % tuple(p.resp_keys))

        slide("""
              For the two kinds of decisions (motion and color), you'll
              be responding using the same two buttons, so each button
              means different things depending on the decision context.

              On motion trials:
                index : %(motion_left)s
                middle: %(motion_right)s

              On color trials:
                index : %(color_left)s
                middle: %(color_right)s

              Again, don't worry too much about memorizing these now.
              """ % button_ass)

        slide("""
              Our aim is to get a precise measurement of how long it takes
              you to make these decisions. Although it is important that you
              answer accurately, we ask that you start making the decision
              as soon as the dots appear and respond right when you have
              made the decision.

              Your response will be accepted as long as the dots are still
              on the screen for that trial, but again, press the correct
              button as soon as you have an answer.
              """)

        slide("""
              Because we are trying to get precise measurements, it's
              important that you pay as close attention as possible during
              the experiment, so you can respond quickly and accurately.

              For the parts that take place outside the scanner, there will
              be frequent breaks to allow you to rest.
              """)

        slide("""
              It's also important that, to minimize eye movements, you keep
              your eyes fixated on the spot in the center of the screen
              throughout  the experiment whenever it is present.

              The fixation spot will turn from black to white very shortly
              before each trial begins to signal that you should get ready.
              """)

        slide("""
              The basic structure of the experiment is as follows.

              The first session (today) is focused on helping you learn
              the task and calibrating the difficulty to your performance.
              """)

        slide("""
              During the initial part, you will learn what each frame pattern
              means and which buttons to press for each decision.

              For this part of the experiment, the stimuli will be presented
              with no noise, and you should focus on learning how to perform
              the task. As you become more comfortable, practice making your
              decisions as fast as you can.

              To help you learn, you'll receive feedback when you make a
              mistake. After a trial where you make an error, the frame
              surrounding the stimulus will flicker a few times to signal
              an incorrect response. Nothing will happen after correct trials.

              Press space to see an example of this negative feedback.
              You can also practice fixating on the central spot.
              """, False)

        frame.draw()
        win.flip()
        cregg.wait_check_quit(1)

        fb_frames = int(p.fb_dur * win.refresh_hz) * 2
        for i in xrange(fb_frames):
            if not i % p.fb_freq:
                frame.flip_phase()
            frame.draw()
            win.flip()

        frame.reset_phase()
        slide("""
              The second task will begin to introduce noise into the stimuli.
              Our goal here is to find a level of noise for the dot motion
              and color so that the two decisions are equally difficult.

              You'll see the level of noise changing over the course of this
              part as we try to dial in the parameters for you.

              To help us calibrate the difficulty, it's important that you
              answer quickly and accurately, just as you will during the
              main experiment.

              You'll also receive feedback during this task.
              """)

        slide("""
              The third and final part you'll perform today will be a practice
              session that is very similar to what you'll be doing inside the
              scanner. You'll no longer be receiving feedback for this part,
              and it's important that you try your best to meet the performance
              criterion to proceed to the scanning session.

              Because scanner time is quite valuable, we can only scan
              participants who are able to perform at a high level of
              speed and accuracy during the practice session.
              """)

        slide("""
              That was a lot of information! The highlights:

              The frame pattern says whether you should make a decision
              about the direction of motion or the dominant color in the
              field of moving dots.

              Start making your decision when the dots come on the screen,
              and respond as quickly as you have an answer.

              Fixate on the central spot whenever it is on the screen.

              Ask the experimenter if you have questions about these or
              any of the other instructions. Good luck!
              """)

        stims["finish"].draw()


def learn(p, win, stims):
    """Blocked trials at full coherence with feedback until learned."""

    # Draw the instructions
    stims["instruct"].draw()

    # Set up the log object
    log_cols = ["block", "block_trial", "context", "cue", "frame_id",
                "motion", "color", "correct", "rt", "response",
                "stim_onset", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Set up the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p, feedback=True)

    # Track the blocks above threshold for each frame
    frame_perf = {k: 0 for k in np.ravel(p.frame_ids)}

    # Set the coherence values
    stims["dots"].motion_coherence = p.coherence
    stims["dots"].color_coherence = p.coherence

    # Main experiment loop
    learned = False
    block = 0
    cue = 0
    with cregg.PresentationLoop(win, p, log=log):
        stim_event.clock.reset()
        while not learned:

            block_acc = []

            context = block % 2
            if not context:
                cue = (cue + 1) % p.frame_per_context

            block_info = dict(block=block, context=context, cue=cue)
            frame_id = p.frame_ids[context][cue]
            stims["frame"].make_active(frame_id)
            stims["frame"].draw()
            win.flip()
            cregg.wait_check_quit(p.iti[0])

            # Loop through the block trials
            for block_trial in xrange(p.n_per_block):

                # Get the feature values for this trial
                motion, color, target = trial_values(p, context)

                # Set up the trial info dict
                t_info = dict(block_trial=block_trial,
                              motion=motion,
                              color=color)
                t_info.update(block_info)

                # Intra-trial interval
                now = stim_event.clock.getTime()
                iti = uniform(*p.iti)
                cue_onset = now + iti
                stims["frame"].draw()
                win.flip()
                cregg.wait_check_quit(iti - p.orient_dur)

                # Stimulus event happens here
                res = stim_event(cue_onset, context, cue, motion, color,
                                 target, frame_with_orient=True)
                t_info.update(res)
                log.add_data(t_info)

                block_acc.append(res["correct"])

            frame_perf[frame_id] += np.mean(block_acc) >= p.perf_thresh
            if (np.array(p.blocks_at_thresh) <= frame_perf.values()).all():
                learned = True
                continue

            if block and not block % p.blocks_bw_break:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()
                cregg.wait_check_quit(p.iti[1])

            block += 1

        stims["finish"].draw()


def staircase(p, win, stims):
    """Find ideal coherence values for each context with a staircase."""

     # Draw the instructions
    stims["instruct"].draw()

    # Set up the log object
    log_cols = ["block", "block_trial",
                "context", "cue", "frame_id",
                "motion_coherence", "color_coherence",
                "motion", "color",
                "correct", "rt", "response",
                "stim_onset", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Set up the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p, feedback=True)
    dots = stims["dots"]

    # Set the coherence values
    dots.motion_coherence = p.starting_coherence
    dots.color_coherence = p.starting_coherence

    log.coherences = [p.starting_coherence] * 2

    # Set up the accuracy trackers
    # The convention is that wrong answers are positive (up)
    # while correct answers are negative (down)
    resp_accs = [0, 0]
    cue = 0

    with cregg.PresentationLoop(win, p, log=log, exit_func=staircase_exit):
        stim_event.clock.reset()
        for block in xrange(p.n_blocks):

            block_trial = 0
            context = block % 2
            if not context:
                cue = (cue + 1) % p.frame_per_context

            block_info = dict(block=block, context=context, cue=cue)
            frame_id = p.frame_ids[context][cue]
            stims["frame"].make_active(frame_id)
            stims["frame"].draw()
            win.flip()
            cregg.wait_check_quit(p.iti[0])

            # Loop through the block trials
            for block_trial in xrange(p.n_per_block):

                # Get the feature values for this trial
                motion, color, target = trial_values(p, context)

                dots.motion_coherence = log.coherences[0]
                dots.color_coherence = log.coherences[1]

                # Set up the trial info dict
                t_info = dict(block_trial=block_trial,
                              motion=motion,
                              color=color)
                t_info.update(dict(zip(["motion_coherence",
                                        "color_coherence"],
                                       log.coherences)))
                t_info.update(block_info)

                # Intra-trial interval
                now = stim_event.clock.getTime()
                iti = uniform(*p.iti)
                cue_onset = now + iti
                stims["frame"].draw()
                win.flip()
                cregg.wait_check_quit(iti - p.orient_dur)

                # Stimulus event happens here
                res = stim_event(cue_onset, context, cue, motion, color,
                                 target, frame_with_orient=True)
                t_info.update(res)
                log.add_data(t_info)

                # Track the accuracies for staircasing
                if block < p.burn_in_blocks:
                    continue

                step = -1 if res["correct"] else 1
                if np.sign(resp_accs[context]) == step:
                    resp_accs[context] += step
                else:
                    resp_accs[context] = step

                # Update the coherence values
                if resp_accs[context] == p.n_up:
                    step_sign = 1
                    resp_accs[context] = 0
                elif resp_accs[context] == -p.n_down:
                    step_sign = -1
                    resp_accs[context] = 0
                else:
                    step_sign = 0
                log.coherences[context] += step_sign * p.step

            if block and not block % p.blocks_bw_break:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()
                cregg.wait_check_quit(p.iti[1])

        stims["finish"].draw()


def staircase_exit(log):
    """Save the final coherence values and report to terminal."""
    coh_file = log.p.coh_file_template % log.p.subject
    cregg.archive_old_version(coh_file)

    df = pd.read_csv(log.fname)
    last_block = df.block.max()
    df = df[df.block >= last_block - log.p.n_calc_blocks]

    coh_dict = dict()
    coh_dict["motion"] = df[df.context == 0].motion_coherence.mean()
    coh_dict["color"] = df[df.context == 1].color_coherence.mean()

    with open(coh_file, "w") as fid:
        json.dump(coh_dict, fid)

    print dedent("""
                 Final coherence:
                   motion: %(motion).2f
                   color: %(color).2f
                 """ % coh_dict)

    print "Accuracy in calculation blocks:"
    grouped = df.groupby(df.context.map({0: "motion", 1: "color"}))
    print grouped.correct.mean()


def practice(p, win, stims):
    """Random (balanced) run-time generated for practicing outside scanner."""

     # Draw the instructions
    stims["instruct"].draw()

    # Find and set this subjects' coherence values
    subject_coherence(p, stims)

    # Set up the log object
    log_cols = ["trial", "context", "cue", "frame_id",
                "motion", "color",
                "correct", "rt", "response",
                "stim_onset", "dropped_frames"]
    log = cregg.DataLog(p, log_cols)

    # Set up the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p)

    # Execute the experimental loop
    with cregg.PresentationLoop(win, p, log=log, exit_func=scan_exit):
        stim_event.clock.reset()
        for trial in xrange(p.n_trials):

            t_info = dict(trial=trial)

            # Intra-trial interval
            now = stim_event.clock.getTime()
            iti = uniform(*p.iti)
            cue_time = now + iti
            stims["fix"].draw()
            win.flip()
            cregg.wait_check_quit(iti - p.orient_dur)

            # Generate stimulus information for this trial
            s_info = dict(
                cue_time=cue_time,
                context=cregg.flip(),
                cue=cregg.flip(),
                motion=cregg.flip(),
                color=cregg.flip(),
            )
            target = [s_info["motion"], s_info["color"]][s_info["context"]]
            s_info["target"] = target
            t_info.update(s_info)

            # Stimulus event happens here
            res = stim_event(**s_info)
            t_info.update(res)
            log.add_data(t_info)

            if trial and not trial % p.trials_bw_break:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()
                cregg.wait_check_quit(p.iti[1])

        stims["finish"].draw()


def trial_values(p, context):
    """Get motion, color, and target value for training."""
    motion = randint(len(p.dot_dirs))
    color = randint(len(p.dot_colors))
    target = [motion, color][context]
    return motion, color, target


def subject_coherence(p, stims):
    """Set the subject-specfic coherece values on the dot object."""
    coh_file = p.coh_file_template % p.subject
    with open(coh_file) as fid:
        coherences = json.load(fid)
    p.__dict__.update(coherences)
    contexts = ["motion", "color"]
    mot_coh, col_coh = [coherences[c] for c in contexts]
    stims["dots"].motion_coherence = mot_coh
    stims["dots"].color_coherence = col_coh


# Stimulus Classes
# ================


class EventEngine(object):
    """Controls execution of a random dot decision trial."""
    def __init__(self, win, stims, p, feedback=False):

        self.win = win
        self.stims = stims
        self.p = p
        self.feedback = feedback
        self.fix = stims["fix"]
        self.frame = stims["frame"]
        self.dots = stims["dots"]
        self.resp_keys = p.resp_keys
        self.orient_dur = p.orient_dur
        self.fix_iti_color = p.fix_iti_color
        self.fix_stim_color = p.fix_stim_color
        self.frame_ids = p.frame_ids
        self.clock = core.Clock()
        self.debug = p.debug
        if feedback:
            self.fb_dur = p.fb_dur
            self.fb_freq = p.fb_freq
        if self.debug:
            self.debug_text = [visual.TextStim(win,
                                               pos=(0.0, -5.0),
                                               height=0.5),
                               visual.TextStim(win,
                                               pos=(0.0, 5.0),
                                               height=0.5)]

    def __call__(self, cue_time, context, cue, motion, color,
                 target, early=False, cue_dur=0, stim=True,
                 frame_with_orient=False):
        """Executes the trial."""

        # Debugging information
        if stim and self.debug:
            dir_name = self.p.dot_dir_names[int(motion)]
            color_name = self.p.dot_color_names[int(color)]
            msg1 = "Motion: %s   Color: %s" % (dir_name, color_name)
            self.debug_text[0].setText(msg1)
            msg2 = ["motion", "color"][context]
            self.debug_text[1].setText(msg2)

        # Set the appropriate frame
        frame_id = self.frame_ids[context][cue]
        self.frame.make_active(frame_id)

        if stim:
            self.dots.direction = self.p.dot_dirs[int(motion)]
            self.dots.color = self.p.dot_colors[int(color)]
            self.dots.new_array()

        # Orient cue
        self.fix.setColor(self.fix_stim_color)
        if frame_with_orient:
            orient_stim = self.frame
        else:
            orient_stim = self.fix
        cregg.precise_wait(self.win, self.clock, cue_time, orient_stim)

        # Early Cue Presentation
        if early:
            self.frame.draw()
            self.win.flip()
            cue_onset_time = self.clock.getTime()
            cregg.wait_check_quit(cue_dur)

        # Main Stimulus Presentation
        if stim:
            dropped_before = self.win.nDroppedFrames
            event.clearEvents()
            stim_frames = int(self.p.stim_dur * self.win.refresh_hz)
            for frame in xrange(stim_frames):
                self.dots.draw()
                self.frame.draw()
                if self.debug:
                    for text in self.debug_text:
                        text.draw()
                self.win.flip()
                if not frame:
                    resp_clock = core.Clock()
                    stim_onset_time = self.clock.getTime()

            motion_signal = np.mean(self.dots.motion_signals)
            color_signal = np.mean(self.dots.color_signals)

            dropped_after = self.win.nDroppedFrames
            dropped_frames = dropped_after - dropped_before

            if not early:
                cue_onset_time = stim_onset_time

            # Response Collection
            response = None
            correct = False
            rt = np.nan
            keys = event.getKeys(timeStamped=resp_clock)
            for key, stamp in keys:
                if key in self.p.quit_keys:
                    print "Subject quit execution"
                    core.quit()
                elif key in self.resp_keys:
                    response = self.resp_keys.index(key)
                    rt = stamp
                    if response == target:
                        correct = True
        else:
            dropped_frames = np.nan
            stim_onset_time = np.nan
            motion_signal = np.nan
            color_signal = np.nan
            response = np.nan
            correct = np.nan
            rt = np.nan

        # Feedback
        if self.feedback and not correct:
            fb_frames = int(self.fb_dur * self.win.refresh_hz)
            for frame in xrange(fb_frames):
                if not frame % self.fb_freq:
                    self.frame.flip_phase()
                self.frame.draw()
                self.win.flip()

            self.frame.reset_phase()

        # Reset fixation color
        self.fix.setColor(self.fix_iti_color)

        # Return event information back to the caller
        result = dict(correct=correct, rt=rt,
                      response=response,
                      frame_id=frame_id,
                      cue_onset=cue_onset_time,
                      stim_onset=stim_onset_time,
                      motion_signal=motion_signal,
                      color_signal=color_signal,
                      dropped_frames=dropped_frames)

        return result


class Frame(object):
    """Square frame around the dot field that serves as a context cute."""
    def __init__(self, win, p, fix=None):

        self.win = win
        self.field_size = p.field_size
        self.frame_width = p.frame_width
        self.texture = p.frame_tex
        self.contrasts = p.frame_contrasts
        self.sfs = p.frame_sfs
        self.phases = p.frame_phases
        self.patterns = p.frame_patterns
        self.fix = fix

        frames = []
        for i in range(2 * p.frame_per_context):
            args = [self.patterns[i], self.contrasts[i],
                    self.sfs[i], self.phases[i]]
            walls = self._make_walls(*args)
            floors = self._make_floors(*args)
            frames.append(walls + floors)
        self.frames = dict(zip(letters[:2 * p.frame_per_context], frames))

    def _make_floors(self, pattern, contrast, sf, phase):
        """Create the PsychoPy objects for top and bottom frame components."""
        ypos = self.field_size / 2
        positions = [(0, -ypos), (0, ypos)]
        length = self.field_size + self.frame_width
        width = self.frame_width
        size_dict = dict(long=[(width, length)], short=[(length, width)])
        size_dict["plaid"] = size_dict["short"] + size_dict["long"]
        ori_dict = dict(long=[90], short=[0],)
        ori_dict["plaid"] = ori_dict["short"] + ori_dict["long"]
        opacity = 0.5 if pattern == "plaid" else 1

        elems = []
        for pos in positions:
            for size, ori in zip(size_dict[pattern], ori_dict[pattern]):
                obj = visual.GratingStim(self.win,
                                         tex=self.texture,
                                         pos=pos,
                                         size=size,
                                         sf=sf,
                                         ori=ori,
                                         phase=phase,
                                         contrast=contrast,
                                         opacity=opacity,
                                         interpolate=True)
                elems.append(obj)
        return elems

    def _make_walls(self, pattern, contrast, sf, phase):
        """Create the PsychoPy objects for left and right frame components."""
        xpos = self.field_size / 2
        positions = [(-xpos, 0), (xpos, 0)]
        length = self.field_size + self.frame_width
        width = self.frame_width
        size_dict = dict(long=[(width, length)], short=[(length, width)])
        size_dict["plaid"] = size_dict["short"] + size_dict["long"]
        ori_dict = dict(short=[90], long=[0])
        ori_dict["plaid"] = ori_dict["short"] + ori_dict["long"]
        opacity = 0.5 if pattern == "plaid" else 1

        elems = []
        for pos in positions:
            for size, ori in zip(size_dict[pattern], ori_dict[pattern]):
                obj = visual.GratingStim(self.win,
                                         tex=self.texture,
                                         pos=pos,
                                         size=size,
                                         sf=sf,
                                         ori=ori,
                                         phase=phase,
                                         contrast=contrast,
                                         opacity=opacity,
                                         interpolate=True)
                elems.append(obj)
        return elems

    def make_active(self, id):
        """Set the frame for the particular task context."""
        self.active_index = letters.index(id)
        self.active_frame = self.frames[id]

    def flip_phase(self):
        """Change the grating phase by half for feedback."""
        for elem in self.active_frame:
            elem.setPhase((elem.phase + .5) % 1)

    def reset_phase(self):
        """Reset the grating phase following feedback."""
        for elem in self.active_frame:
            elem.setPhase(self.phases[self.active_index])

    def draw(self):
        """Draw the components of the active frame to the screen."""
        for elem in self.active_frame:
            elem.draw()
        if self.fix is not None:
            self.fix.draw()


class Dots(object):
    """Random dot field stimulus."""
    def __init__(self, win, p):

        # Move some info from params into self
        self.subframes = p.dot_subframes
        self.ndots = p.dot_count
        self.speed = p.dot_speed / win.refresh_hz
        self.colors = np.array(p.dot_colors)
        self.field_size = p.field_size - p.frame_width

        # Initialize the Psychopy object
        dot_shape = None if p.dot_shape == "square" else p.dot_shape
        self.dots = visual.ElementArrayStim(win, "deg",
                                            fieldShape="square",
                                            fieldSize=p.field_size,
                                            nElements=p.dot_count,
                                            sizes=p.dot_size,
                                            elementMask=dot_shape,
                                            colors=np.ones((p.dot_count, 3)),
                                            elementTex=None,
                                            )

        # Use a cycle to control which set of dots is getting drawn
        self._dot_cycle = itertools.cycle(range(self.subframes))

    def new_array(self):
        """Initialize a new set of evently-distributed dot positions."""
        half_field = self.field_size / 2
        locs = np.linspace(-half_field, half_field, 5)
        while True:
            xys = np.random.uniform(-half_field, half_field,
                                    size=(self.subframes, 2, self.ndots))
            ps = np.zeros(self.subframes)
            for i, field in enumerate(xys):
                table, _, _ = np.histogram2d(*field, bins=(locs, locs))
                if not table.all():
                    continue
                _, p, _, _ = stats.chi2_contingency(table)
                ps[i] = p
            if (ps > 0.1).all():
                break

        self._xys = xys.transpose(2, 1, 0)
        self.motion_signals = []
        self.color_signals = []

    def _update_positions(self):
        """Move some dots in one direction and redraw others randomly."""
        active_dots = self._xys[..., self._dot_set]

        # This is how we get an average coherence with random signal
        signal = np.random.uniform(size=self.ndots) < self.motion_coherence
        self.motion_signals.append(signal.mean())

        # Update the positions of the signal dots
        dir = np.deg2rad(self.direction)
        active_dots[signal, 0] += self.speed * self.subframes * np.cos(dir)
        active_dots[signal, 1] += self.speed * self.subframes * np.sin(dir)

        # Find new random positions for the noise dots
        noise = ~signal
        half_field = self.field_size / 2
        active_dots[noise] = np.random.uniform(-half_field, half_field,
                                               size=(noise.sum(), 2))

        # Deal with signal dots going out of bounds
        self._wrap_around(active_dots)

        # Update the positions in the psychopy object and store in this object
        self.dots.setXYs(active_dots)

    def _update_colors(self):
        """Set dot colors using the stored coherence value."""
        signal = np.random.uniform(size=self.ndots) < self.color_coherence
        self.color_signals.append(signal.mean())
        rgb = np.zeros((self.ndots, 3))
        rgb[signal] = self.color

        noise = ~signal
        noise_colors = (np.random.uniform(size=noise.sum()) < 0.5).astype(int)
        rgb[noise] = self.colors[noise_colors]
        rgb = rgb * 2 - 1
        self.dots.setColors(rgb)

    def _wrap_around(self, dots):
        """Redraw dots off the FOV on the other side."""
        half_field = self.field_size / 2
        out_of_bounds = np.abs(dots) > half_field
        pos = dots[out_of_bounds]
        dots[out_of_bounds] = -1 * (pos / np.abs(pos)) * half_field

    def draw(self):
        """Update the dot positions based on direction and draw."""
        self._dot_set = self._dot_cycle.next()
        self._update_positions()
        self._update_colors()

        self.dots.draw()


if __name__ == "__main__":
    main(sys.argv[1:])
