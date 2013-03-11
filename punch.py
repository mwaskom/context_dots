"""Main script for the parametric uncertainty project.

See project README for more information about this experiment and
the way it is controlled by this program.

"""
from __future__ import division
import sys
import colorsys
import json
from string import letters
from textwrap import dedent
import pandas as pd
import numpy as np
from numpy.random import randint, uniform
from scipy import stats
from psychopy import visual, core, event
import tools


def main(arglist):

    # Get the experiment paramters
    mode = arglist.pop(0)
    p = tools.Params("punch_%s" % mode)
    p.set_by_cmdline(arglist)

    # Assign the frame identities randomly over subjects
    state = tools.subject_specific_state(p.subject)
    frame_ids = state.permutation(list(letters[:2 * p.frame_per_context]))
    p.frame_ids = frame_ids.reshape(2, -1).tolist()

    # Open up the stimulus window
    win = tools.launch_window(p)

    # Set up the stimulus objects
    fix = visual.GratingStim(win, tex=None, mask=p.fix_shape,
                             color=p.fix_isi_color, size=p.fix_size)

    instruct_text = dedent(p.instruct_text)
    instruct = tools.WaitText(win, instruct_text,
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
        take_break = tools.WaitText(win, break_text,
                                    height=p.break_text_size,
                                    advance_keys=p.wait_keys,
                                    quit_keys=p.quit_keys)
        stims["break"] = take_break

    if hasattr(p, "finish_text"):
        finish_text = dedent(p.finish_text)
        finish_run = tools.WaitText(win, finish_text,
                                    height=p.break_text_size,
                                    advance_keys=p.wait_keys,
                                    quit_keys=p.quit_keys)
        stims["finish"] = finish_run

    # Excecute the experiment function
    globals()[mode](p, win, stims)


# Experiment Functions
# ====================


def behav(p, win, stims):
    """Behavioral experiment."""

    # Max the screen brightness
    tools.max_brightness(p.monitor_name)

    # Get the design
    d = tools.load_design_csv(p)
    p.n_trials = len(d)

    # Randomize for this execution
    d["sorter"] = np.random.rand(p.n_trials)
    d.sort("sorter", inplace=True)
    d.index = range(p.n_trials)

    # Draw the instructions
    stims["instruct"].draw()

    # Set up the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p)

    # Determine the coherence for this subject
    coh_file = p.coh_file_template % p.subject
    with open(coh_file) as fid:
        coherences = json.load(fid)
    p.__dict__.update(coherences)
    coh_vals = [coherences["dot_%s_coh" % c] for c in ["mot", "col"]]
    stims["dots"].new_signals(*coh_vals)

    # Set up the log files
    d_cols = list(d.columns)
    log_cols = d_cols + ["frame_id", "cue_dur", "response", "rt",
                         "context_switch", "frame_switch",
                         "color_switch", "motion_switch",
                         "correct", "isi", "onset_time", "dropped_frames"]
    log = tools.DataLog(p, log_cols)
    log.dots = dict(mot_signal=stims["dots"].mot_signal,
                    col_signal=stims["dots"].col_signal,
                    dirs=np.zeros((p.n_trials, p.dot_count)),
                    hues=np.zeros((p.n_trials, p.dot_count)))

    # Execute the experiment
    with tools.PresentationLoop(win, log, behav_exit):
        stim_event.clock.reset()
        for t in xrange(p.n_trials):

            # Get the info for this trial
            t_info = {k: d[k][t] for k in d_cols}

            context = d.context[t]
            cue = d.cue[t]
            frame_id = p.frame_ids[context][cue]
            stims["frame"].make_active(frame_id)
            t_info["frame_id"] = frame_id

            early = bool(d.early[t])
            cue_dur = uniform(*p.cue_dur) if early else None
            t_info["cue_dur"] = cue_dur

            motion = d.motion[t]
            color = d.color[t]
            target = [motion, color][context]

            # Figure out if switches have happened and log
            if not t:
                t_info.update(
                    {"%s_switch" % k: True for k in ["context", "frame",
                                                     "motion", "color"]})
            else:
                t_info["context_switch"] = context != d.context[t - 1]
                t_info["motion_switch"] = motion != d.motion[t - 1]
                t_info["color_switch"] = color != d.color[t - 1]
                t_info["frame_switch"] = (context != d.context[t - 1] or
                                          cue != d.cue[t - 1])

            # Pre-stim fixation
            isi = uniform(*p.isi)
            t_info["isi"] = isi
            stims["fix"].draw()
            win.flip()
            tools.wait_check_quit(isi)

            # The stimulus event actually happens here
            res = stim_event(context, motion, color,
                             target, early, cue_dur)
            t_info.update(res)
            log.add_data(t_info)
            log.dots["dirs"][t] = stims["dots"].dirs
            log.dots["hues"][t] = stims["dots"].hues

            # Every n trials, let the subject take a quick break
            if t and not t % p.trials_bw_breaks:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()
                tools.wait_check_quit(p.isi[1])

        stims["finish"].draw()


def behav_exit(log):
    """Gets executed at the end of a behavioral run."""
    # Save the dot stimulus data to a npz archive
    dots_fname = log.fname.strip(".csv")
    np.savez(dots_fname, **log.dots)

    # Read in the data file and print some performance information
    run_df = pd.read_csv(log.fname)
    if not len(run_df):
        return
    print "Overall Accuracy: %.2f" % run_df.correct.mean()
    print run_df.groupby("context").correct.mean()
    print "Average RT: %.2f" % run_df.rt.mean()
    print run_df.groupby("context").rt.mean()


def train(p, win, stims):
    """Training for behavioral and fMRI experiments."""

    # Max the screen brightness
    tools.max_brightness(p.monitor_name)

    # Draw the instructions
    stims["instruct"].draw()

    # Set up the log object
    log_cols = ["block", "learned", "settled",
                "context", "cue", "frame_id",
                "motion", "color", "motion_coh", "color_coh",
                "correct", "rt", "response", "onset_time",
                "dropped_frames"]
    log = tools.DataLog(p, log_cols)

    # Set up the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p, feedback=True)

    # Set up some variables used in logic below
    # that assess how training is going
    learned = False
    settled = False
    trained = False
    coherences = [1, 1]
    context_good = [0, 0]
    settle_accs = [0, 0]
    motion_med = []
    color_adj = -1
    col_reversals = 0

    # Main experiment loop
    block = 0
    cue = 0
    with tools.PresentationLoop(win, log, train_exit):
        stim_event.clock.reset()
        while not trained:

            # Reset the info for this block
            block_rts = []
            block_acc = []

            context = block % 2
            if not context:
                cue = (cue + 1) % p.frame_per_context
            frame_id = p.frame_ids[context][cue]
            stims["frame"].make_active(frame_id)
            stims["dots"].new_signals(*coherences)

            block_info = dict(block=block, context=context,
                              cue=cue, frame_id=frame_id,
                              learned=learned, settled=settled,
                              motion_coh=coherences[0],
                              color_coh=coherences[1])

            # Draw the frame to start the block
            stims["frame"].draw()
            win.flip()
            tools.wait_check_quit(p.isi[1])

            # Loop through the block trials
            for trial in xrange(p.n_per_block):

                # Get the feature values for this trial
                motion = randint(len(p.dot_dirs))
                color = randint(len(p.dot_colors))
                target = color if context else motion

                # Set up the trial info drink
                t_info = dict(motion=motion, color=color)
                t_info.update(block_info)

                # Stimulus event happens here
                res = stim_event(context, motion, color, target)
                t_info.update(res)
                log.add_data(t_info)
                block_rts.append(res["rt"])
                block_acc.append(res["correct"])

                # Draw the frame for the inter-trial interval
                stims["frame"].draw()
                win.flip()
                tools.wait_check_quit(uniform(*p.isi))

            # Every n trials, let the subject take a quick break
            if block and not block % p.blocks_bw_break:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()
                tools.wait_check_quit(p.isi[1])

            block += 1

            # Handle the main training logic in three three stages:
            # First we have full-coherence blocks until subjects
            # have learned the cue-context and stimulus-response mappings
            # Then, we gradually decrease the coherence until we hit the
            # target for motion blocks. Finally, we staircase the color
            # coherence around a bit to try and match RTs.

            if not learned:
                # Update the accuracy history
                if np.mean(block_acc) >= p.full_coh_thresh:
                    context_good[context] += 1
                # Check if we've reached the criterion for
                # full-coherence learning
                if min(context_good) >= p.at_thresh_blocks:
                    learned = True
                continue

            if not settled:
                # If this is a color block and we are above the
                # accuracy threshold for the settle period on this and
                # the previous motion block, decrement the coherence
                settle_accs[context] = np.mean(block_acc)
                if context and min(settle_accs) >= p.settle_thresh:
                    coherences = [c - p.settle_slope for c in coherences]
                    # Check if we've hit the target for motion coherence
                    if abs(coherences[0] - p.motion_coh_target) < 1e-6:
                        settled = True
                continue

            if not trained:
                # Staircase the color coherence to try and match
                # reaction times to the motion coherence
                if context:
                    color_med = stats.nanmedian(block_rts)
                    if color_adj == -1:
                        # When lowering color coherence, reverse
                        # if motion is faster (easier) than color
                        reverse = color_med > motion_med
                    else:
                        # The converse is the case when we are
                        # stepping up the color coherence
                        reverse = color_med < motion_med

                    if reverse:
                        color_adj *= -1
                        col_reversals += 1

                    try:
                        # Index the list of step sizes by the number
                        # of reversals we've made. If we've made the
                        # desired number of reversals, this will
                        # raise an IndexError and we'll catch that
                        # exception to say we are trained
                        step_size = p.reversal_steps[col_reversals]
                        step = color_adj * step_size
                        coherences[1] += step
                    except IndexError:
                        trained = True
                else:
                    motion_med = stats.nanmedian(block_rts)

        print "Final color coherence: %.2f" % coherences[1]
        coherences = dict(zip(["dot_mot_coh", "dot_col_coh"], coherences))
        with open(p.coh_file_template % p.subject, "w") as fid:
            json.dump(coherences, fid)
        stims["finish"].draw()


def train_exit(log):
    """Gets executed at the end of training."""
    df = pd.read_csv(log.fname)
    if not len(df):
        return
    print "Training took %d blocks" % df.block.unique().size


def demo(p, win, stims):
    """Brief demonstration of stimuli before training."""

    with tools.PresentationLoop(win):
        frame = stims["frame"]
        stims["dots"].new_signals(*[p.motion_coh_target] * 2)

        stim_event = EventEngine(win, stims, p)
        frame.make_active("a")

        tools.wait_and_listen("space")

        stim_event(0, 0, 0, 0)
        win.flip()
        tools.wait_and_listen("space")

        stim_event(0, 0, 0, 0)
        win.flip()
        tools.wait_and_listen("space")

        for context in [0, 1]:
            for cue in range(2):
                id = [["a", "b"], ["c", "d"]][context][cue]
                frame.make_active(id)
                frame.draw()
                win.flip()
                tools.wait_and_listen("space")

        for context in [0, 1]:
            for cue in range(2):
                id = [["a", "b"], ["c", "d"]][context][cue]
                frame.make_active(id)
                frame.draw()
                win.flip()
                tools.wait_and_listen("space")
                for refresh in xrange(p.fb_dur * 60):
                    if not refresh % p.fb_freq:
                        frame.flip_phase()
                    frame.draw()
                    win.flip()
                tools.wait_and_listen("space")


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
        self.fix_orient_dur = p.fix_orient_dur
        self.fix_isi_color = p.fix_isi_color
        self.fix_stim_color = p.fix_stim_color
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

    def __call__(self, context, motion, color, target,
                 early=False, cue_dur=0):
        """Executes the trial."""

        self.dots.new_positions()

        # Debugging information
        if self.debug:
            motion_deg = self.p.dot_dirs[motion]
            color_name = self.p.dot_colors[color]
            msg1 = "Motion: %d   Color: %s" % (motion_deg, color_name)
            self.debug_text[0].setText(msg1)
            msg2 = ["motion", "color"][context]
            self.debug_text[1].setText(msg2)

        self.dots.new_colors(color)
        self.dots.new_directions(motion)

        # Reorient cue
        self.fix.setColor(self.fix_stim_color)
        self.fix.draw()
        self.win.flip()
        tools.wait_check_quit(self.fix_orient_dur)

        # Early Cue Presentation
        if early:
            self.frame.draw()
            self.win.flip()
            tools.wait_check_quit(cue_dur)

        # Main Stimulus Presentation
        dropped_before = self.win.nDroppedFrames
        event.clearEvents()
        for frame in xrange(self.p.stim_dur * 60):
            self.dots.draw()
            self.frame.draw()
            if self.debug:
                for text in self.debug_text:
                    text.draw()
            self.win.flip()
            if not frame:
                resp_clock = core.Clock()
                onset_time = self.clock.getTime()
        dropped_after = self.win.nDroppedFrames
        dropped_frames = dropped_after - dropped_before

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

        # Feedback
        if self.feedback and not correct:
            for frame in xrange(self.fb_dur * 60):
                if not frame % self.fb_freq:
                    self.frame.flip_phase()
                self.frame.draw()
                self.win.flip()

            self.frame.reset_phase()

        # Reset fixation color
        self.fix.setColor(self.fix_isi_color)

        result = dict(correct=correct, rt=rt, response=response,
                      onset_time=onset_time, dropped_frames=dropped_frames)

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
                                         opacity=opacity)
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
                                         opacity=opacity)
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

        self.speed = p.dot_speed / 60
        self.colors = p.dot_colors
        self.dot_hues = p.dot_hues
        self.dot_sat = p.dot_sat
        self.dot_val = p.dot_val
        self.dot_dirs = p.dot_dirs
        self.field_size = p.field_size - p.frame_width
        self.ndots = p.dot_count
        self.dot_life_mean = p.dot_life_mean
        self.dot_life_std = p.dot_life_std
        self.dimension = len(p.dot_colors)
        assert self.dimension == len(p.dot_dirs)

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

        self.dot_life = np.round(np.random.normal(p.dot_life_mean,
                                                  p.dot_life_std,
                                                  size=p.dot_count))

    def new_signals(self, mot_coh, col_coh):
        """Decide which dots carry signal based on context coherence."""
        mot_signal = np.zeros(self.ndots, bool)
        col_signal = np.zeros(self.ndots, bool)

        n_col_signal = col_coh * self.ndots
        col_signal[:n_col_signal] = True

        n_mot_signal_a = mot_coh * n_col_signal
        n_mot_signal_b = mot_coh * (self.ndots - n_col_signal)
        mot_signal[:n_mot_signal_a] = True
        mot_signal[n_col_signal:(n_col_signal + n_mot_signal_b)] = True

        self.col_signal = col_signal
        self.mot_signal = mot_signal

    def new_colors(self, target_color):
        """Set the hues for the signal and noise dots."""
        hues = np.random.uniform(size=self.ndots)
        t_hue = self.dot_hues[target_color]
        hues[self.col_signal] = t_hue
        self.hues = hues
        s = self.dot_sat
        v = self.dot_val
        colors = np.array([colorsys.hsv_to_rgb(h, s, v) for h in hues])
        colors = colors * 2 - 1
        self.dots.setColors(colors)

    def new_directions(self, target_dir):
        """Set the directions for the signal and noise dots."""
        dirs = np.random.uniform(size=self.ndots) * 2 * np.pi
        dirs[self.mot_signal] = np.deg2rad(self.dot_dirs[target_dir])
        self.dirs = dirs

    def new_positions(self, mask=None):
        """Set new positions for all, or a subset, of dots."""
        if mask is None:
            new_size = (self.ndots, 2)
            mask = np.ones((self.ndots, 2), bool)
        else:
            new_size = (mask.sum(), 2)

        half_field = self.field_size / 2
        new_xys = np.random.uniform(-half_field, half_field, new_size)

        xys = self.dots.xys
        if mask.all():
            new_xys = new_xys.ravel()
        xys[mask] = new_xys
        self.dots.setXYs(xys)

    def draw(self):
        """Update the dot positions based on direction and draw."""
        xys = self.dots.xys
        xys[:, 0] += self.speed * np.cos(self.dirs)
        xys[:, 1] += self.speed * np.sin(self.dirs)
        bound = (self.field_size / 2)
        self.dots.setXYs(xys)
        out_of_bounds = np.any(np.abs(xys) > bound, axis=1)
        self.new_positions(out_of_bounds)
        self.dot_life -= 1
        dead_dots = self.dot_life < 0
        self.new_positions(dead_dots)
        self.dot_life[dead_dots] = np.round(
            np.random.normal(self.dot_life_mean,
                             self.dot_life_std,
                             size=self.ndots))

        self.dots.draw()


if __name__ == "__main__":
    main(sys.argv[1:])
