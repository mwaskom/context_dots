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
from husl import husl_to_rgb
import pandas as pd
import numpy as np
from numpy.random import randint, uniform
from scipy import stats
from psychopy import visual, core, event
import tools


def main(arglist):

    # Get the experiment paramters
    mode = arglist.pop(0)
    p = tools.Params(mode)
    p.set_by_cmdline(arglist)

    # Assign the frame identities randomly over subjects
    state = tools.subject_specific_state(p.subject)
    frame_ids = state.permutation(list(letters[:2 * p.frame_per_context]))
    p.frame_ids = frame_ids.reshape(2, -1).tolist()

    # Open up the stimulus window
    win = tools.launch_window(p)
    win.refresh_rate = 60  # TODO FIX

    # Set up the stimulus objects
    fix = visual.GratingStim(win, tex=None,
                             mask=p.fix_shape, interpolate=True,
                             color=p.fix_iti_color, size=p.fix_size)

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


def scan(p, win, stims):
    """Neuroimaging experiment."""

    # Max the screen brightness
    tools.max_brightness(p.monitor_name)

    # Get the design
    d = pd.read_csv(p.design_file)
    d = d[d.run == p.run]
    d.set_index("run_trial", inplace=True)

    # Draw the instructions
    stims["instruct"].draw()

    # Set up the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p)

    # Determine the coherence for this subject
    coh_file = p.coh_file_template % p.subject
    with open(coh_file) as fid:
        coherences = json.load(fid)
    p.__dict__.update(coherences)
    mot_coh, col_coh = [coherences["dot_%s_coh" % c] for c in ["mot", "col"]]
    stims["dots"].motion_coherence = mot_coh
    stims["dots"].color_coherence = col_coh

    # Set up the log files
    d_cols = list(d.columns)
    log_cols = d_cols + ["frame_id", "cue_dur", "response", "rt",
                         "context_switch", "frame_switch",
                         "color_switch", "motion_switch",
                         "correct", "onset_time", "dropped_frames"]
    log = tools.DataLog(p, log_cols)

    # Execute the experiment
    with tools.PresentationLoop(win, log, behav_exit):
        stim_event.clock.reset()
        for t in xrange(p.trials_per_run):

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

            stim = bool(d.stim[t])

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
            stims["fix"].draw()
            win.flip()
            tools.wait_check_quit(d.iti[t] * p.tr)

            # The stimulus event actually happens here
            res = stim_event(context, motion, color,
                             target, early, cue_dur, stim)
            t_info.update(res)
            log.add_data(t_info)

        stims["finish"].draw()


def behav_exit(log):
    """Gets executed at the end of a behavioral run."""
    # Save the dot stimulus data to a npz archive
    return
    dots_fname = log.fname.strip(".csv")
    np.savez(dots_fname)

    # Read in the data file and print some performance information
    run_df = pd.read_csv(log.fname)
    if not len(run_df):
        return
    print "Overall Accuracy: %.2f" % run_df.correct.dropna().mean()
    print run_df.groupby("context").correct.dropna().mean()
    print "Average RT: %.2f" % run_df.rt.dropna().mean()
    print run_df.groupby("context").rt.dropna().mean()


def train(p, win, stims):
    """Training for behavioral and fMRI experiments."""

    # Max the screen brightness
    tools.max_brightness(p.monitor_name)

    # Draw the instructions
    stims["instruct"].draw()

    # Got a local reference to the dots object
    dots = stims["dots"]

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
            mot_coh, col_coh = coherences
            dots.motion_coherence = mot_coh
            dots.color_coherence = col_coh

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
                res = stim_event(context, motion, color, target,
                                 frame_with_orient=True)
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
                        coherences[1] = max(p.color_coh_floor, coherences[1])
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
                for refresh in xrange(p.fb_dur * win.refresh_rate):
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
        self.fix_iti_color = p.fix_iti_color
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
                 early=False, cue_dur=0, stim=True,
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

        if stim:
            self.dots.direction = self.p.dot_dirs[int(motion)]
            self.dots.color = self.p.dot_colors[int(color)]
            self.dots.hue = self.p.dot_hues[int(color)]
            self.dots.new_array()

        # Orient cue
        self.fix.setColor(self.fix_stim_color)
        self.fix.draw()
        if frame_with_orient:
            self.frame.draw()
        self.win.flip()
        tools.wait_check_quit(self.fix_orient_dur)

        # Early Cue Presentation
        if early:
            self.frame.draw()
            self.win.flip()
            tools.wait_check_quit(cue_dur)

        # Main Stimulus Presentation
        if stim:
            dropped_before = self.win.nDroppedFrames
            event.clearEvents()
            for frame in xrange(self.p.stim_dur * self.win.refresh_rate):
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
        else:
            dropped_frames = np.nan
            onset_time = np.nan
            response = np.nan
            correct = np.nan
            rt = np.nan

        # Feedback
        if self.feedback and not correct:
            for frame in xrange(self.fb_dur * self.win.refresh_rate):
                if not frame % self.fb_freq:
                    self.frame.flip_phase()
                self.frame.draw()
                self.win.flip()

            self.frame.reset_phase()

        # Reset fixation color
        self.fix.setColor(self.fix_iti_color)

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
        self.ndots = p.dot_count
        self.speed = p.dot_speed / win.refresh_rate
        self.colors = np.array(p.dot_colors)
        self.saturation = p.dot_saturation
        self.lightness = p.dot_lightness
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
        self._dot_cycle = itertools.cycle(range(3))

    def new_array(self):

        half_field = self.field_size / 2
        locs = np.linspace(-half_field, half_field, 5)
        while True:
            xys = np.random.uniform(-half_field, half_field,
                                    size=(3, 2, self.ndots))
            ps = np.zeros(3)
            for i, field in enumerate(xys):
                table, _, _ = np.histogram2d(*field, bins=(locs, locs))
                if not table.all():
                    continue
                _, p, _, _ = stats.chi2_contingency(table)
                ps[i] = p
            if (ps > 0.1).all():
                break

        self._xys = xys.transpose(2, 1, 0)

    def _update_positions(self):
        """Move some dots in one direction and redraw others randomly."""
        active_dots = self._xys[..., self._dot_set]

        # This is how we get an average coherence with random signal
        signal = np.random.uniform(size=self.ndots) < self.motion_coherence

        # Update the positions of the signal dots
        dir = np.deg2rad(self.direction)
        active_dots[signal, 0] += self.speed * 3 * np.cos(dir)
        active_dots[signal, 1] += self.speed * 3 * np.sin(dir)

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
        rgb = np.zeros((self.ndots, 3))
        rgb[signal] = self.color

        noise = ~signal
        noise_colors = (np.random.uniform(size=noise.sum()) < 0.5).astype(int)
        rgb[noise] = self.colors[noise_colors]
        rgb = rgb * 2 - 1
        self.dots.setColors(rgb)

    def _update_colors_off(self):
        """Set dot colors using the stored coherence value."""
        signal = np.random.uniform(size=self.ndots) < self.color_coherence
        hues = np.random.randint(360, size=self.ndots).astype(float)
        hues[signal] = self.hue
        s = float(self.saturation)
        l = float(self.lightness)
        rgb = np.array([husl_to_rgb(h, s, l) for h in hues])
        rgb[rgb < 0] = 0
        rgb[rgb > 1] = 1
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
