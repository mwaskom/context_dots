from __future__ import division
import sys
import colorsys
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

    # Open up the stimulus window
    win = tools.launch_window(p)

    # Set up the stimulus objects
    fix = visual.GratingStim(win, tex=None, mask=p.fix_shape,
                             color=p.fix_color, size=p.fix_size)

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


def behav(p, win, stims):

    # Max the screen brightness
    tools.max_brightness(p.monitor_name)

    # Get the design
    d = tools.load_design_csv(p)
    p.n_trials = len(d)

    # Draw the instructions
    stims["instruct"].draw()

    # Set up the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p)

    # Set up the log files
    d_cols = list(d.columns)
    log_cols = d_cols + ["cue_dur", "response", "rt", "correct",
                         "isi", "onset_time", "dropped_frames"]
    log = tools.DataLog(p, log_cols)

    # Execute the experiment
    with tools.PresentationLoop(win, log, behav_summary):
        stim_event.clock.reset()
        for t in xrange(p.n_trials):

            # Get the info for this trial
            t_info = {k: d[k][t] for k in d_cols}

            context = d.context[t]
            stims["frame"].set_context(context)

            early = bool(d.early[t])
            cue_dur = uniform(*p.cue_dur) if early else None
            t_info["cue_dur"] = cue_dur

            motion = d.motion[t]
            color = d.color[t]
            target = [motion, color][context]

            # Pre-stim fixation
            isi = uniform(*p.isi)
            t_info["isi"] = isi
            stims["fix"].draw()
            win.flip()
            tools.wait_check_quit(uniform(*p.iti))

            # The stimulus event actually happens here
            res = stim_event(context, motion, color,
                             target, early, cue_dur)
            t_info.update(res)
            log.add_data(t_info)

            # Every n trials, let the subject take a quick break
            if t and not t % p.trials_bw_breaks:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()
                tools.wait_check_quit(p.iti[1])

        stims["finish"].draw()


def train(p, win, stims):

    # Max the screen brightness
    tools.max_brightness(p.monitor_name)

    # Draw the instructions
    stims["instruct"].draw()

    # Set up the log object
    log_cols = ["block", "learned", "settled", "context",
                "motion", "color", "motion_coh", "color_coh",
                "correct", "rt", "response", "onset_time",
                "dropped_frames"]
    log = tools.DataLog(p, log_cols)

    # Setup the object to control stimulus presentation
    stim_event = EventEngine(win, stims, p, feedback=True)

    learned = False
    settled = False
    trained = False
    coherences = [1, 1]
    context_good = [0, 0]
    motion_rts = []
    color_adj = -1
    color_reversals = 0

    block = 0
    with tools.PresentationLoop(win, log, train_summary):
        stim_event.clock.reset()
        while not trained:

            block_rts = []
            block_acc = []

            context = block % 2
            stims["frame"].set_context(context)
            stims["dots"].new_signals(*coherences)

            block_info = dict(block=block, context=context,
                              learned=learned, settled=settled,
                              motion_coh=coherences[0],
                              color_coh=coherences[1])

            stims["frame"].draw()
            win.flip()
            tools.wait_check_quit(p.iti[1])

            for trial in xrange(p.n_per_block):

                motion = randint(len(p.dot_dirs))
                color = randint(len(p.dot_colors))
                target = color if context else motion

                t_info = dict(motion=motion, color=color)
                t_info.update(block_info)

                res = stim_event(context, motion, color, target)
                t_info.update(res)
                log.add_data(t_info)

                block_rts.append(res["rt"])
                block_acc.append(res["correct"])

                stims["frame"].draw()
                win.flip()
                tools.wait_check_quit(uniform(*p.iti))

            # Every n trials, let the subject take a quick break
            if block and not block % p.blocks_bw_break:
                stims["break"].draw()
                stims["fix"].draw()
                win.flip()
                tools.wait_check_quit(p.iti[1])

            block += 1

            # Update the training information
            if learned:
                if settled:
                    if context:
                        motion_mean = stats.nanmean(motion_rts)
                        block_med = stats.nanmedian(block_rts)
                        if color_adj == -1:
                            reverse = block_med > motion_mean
                        else:
                            reverse = block_med < motion_mean
                        if reverse:
                            color_adj *= -1
                            color_reversals += 1
                        coherences[1] += color_adj * p.color_coh_step
                        if color_reversals > p.color_coh_reversals:
                            trained = True
                    else:
                        motion_rts.append(stats.nanmedian(block_rts))
                else:
                    if context:
                        coherences = [c - p.settle_slope for c in coherences]
                        if abs(coherences[0] - p.motion_coh_target) < 1e-6:
                            settled = True
            else:
                if np.mean(block_acc) >= p.full_coh_thresh:
                    context_good[context] += 1
                if all([g >= p.at_thresh_blocks for g in context_good]):
                    learned = True

        stims["finish"].draw()
        print "Final color coherence: %.2f" % coherences[1]


def demo(p, win, stims):

    frame = stims["frame"]
    stims["dots"].new_signals(*[p.motion_coh_target] * 2)

    stim_event = EventEngine(win, stims, p)
    frame.set_context(0)

    tools.wait_and_listen("space")

    stim_event(0, 0, 0, 0)
    win.flip()
    tools.wait_and_listen("space")

    stim_event(0, 0, 0, 0)
    win.flip()
    tools.wait_and_listen("space")

    for context in [0, 1]:
        frame.set_context(context)
        frame.draw()
        win.flip()
        tools.wait_and_listen("space")

    for context in [0, 1]:
        frame.set_context(context)
        frame.draw()
        win.flip()
        tools.wait_and_listen("space")
        for refresh in xrange(p.fb_dur * 60):
            if not refresh % p.fb_freq:
                frame.flip_phase()
            frame.draw()
            win.flip()
        tools.wait_and_listen("space")


class EventEngine(object):

    def __init__(self, win, stims, p, feedback=False):

        self.win = win
        self.stims = stims
        self.p = p
        self.feedback = feedback
        self.fix = stims["fix"]
        self.frame = stims["frame"]
        self.dots = stims["dots"]
        self.resp_keys = p.resp_keys
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
            if key in ["q", "escape"]:
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

        result = dict(correct=correct, rt=rt, response=response,
                      onset_time=onset_time, dropped_frames=dropped_frames)

        return result


class Frame(object):

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

    def set_context(self, context):

        w = self.frame_width
        sizes = [(self.field_size + w, w),
                 (w, self.field_size + w)]
        for i in range(2):
            self.lr[i].setOri([0, 90][context])
            self.tb[i].setOri([90, 0][context])
            for sides in ["lr", "tb"]:
                obj = getattr(self, sides)
                obj[i].setSize(sizes[context])
                obj[i].setSF(self.sf_list[context])

    def set_active(self, id):

        self.active_index = letters.index(id)
        self.active_frame = self.frames[id]

    def flip_phase(self):

        for elem in self.active_frame:
            elem.setPhase((elem.phase + .5) % 1)

    def reset_phase(self):

        for elem in self.active_frame:
            elem.setPhase(self.phases[self.active_index])

    def draw(self):

        for elem in self.active_frame:
            elem.draw()
        if self.fix is not None:
            self.fix.draw()


class Dots(object):

    def __init__(self, win, p):

        self.speed = p.dot_speed / 60
        self.colors = p.dot_colors
        self.dot_hues = p.dot_hues
        self.dot_sat = p.dot_sat
        self.dot_val = p.dot_val
        self.dirs = p.dot_dirs
        self.field_size = p.field_size
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

        self.new_signals(p.dot_mot_coh, p.dot_col_coh)

        self.dot_life = np.round(np.random.normal(p.dot_life_mean,
                                                  p.dot_life_std,
                                                  size=p.dot_count))

    def new_signals(self, mot_coh, col_coh):

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

        hues = np.random.uniform(size=self.ndots)
        t_hue = self.dot_hues[target_color]
        hues[self.col_signal] = t_hue
        s = self.dot_sat
        v = self.dot_val
        colors = np.array([colorsys.hsv_to_rgb(h, s, v) for h in hues])
        colors = colors * 2 - 1
        self.dots.setColors(colors)

    def new_directions(self, target_dir):

        dirs = np.random.uniform(size=self.ndots) * 2 * np.pi
        dirs[self.mot_signal] = np.deg2rad(self.dirs[target_dir])
        self._directions = dirs

    def new_positions(self, mask=None):

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

        xys = self.dots.xys
        xys[:, 0] += self.speed * np.cos(self._directions)
        xys[:, 1] += self.speed * np.sin(self._directions)
        bound = self.field_size / 2
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


def behav_summary(log):
    """Gets executed at the end of a behavioral run."""
    run_df = pd.read_csv(log.fname)
    print "Overall Accuracy: %.2f" % run_df.correct.mean()
    print run_df.groupby("context").correct.mean()
    print "Average RT: %.2f" % run_df.correct.mean()
    print run_df.groupby("context").rt.mean()


def train_summary(log):
    """Gets executed at the end of training."""
    df = pd.read_csv(log.fname)
    print "Training took %d blocks" % df.block.unique().size

if __name__ == "__main__":
    main(sys.argv[1:])
