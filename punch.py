from __future__ import division
import sys
import os.path as op
from textwrap import dedent
from string import letters
import colorsys
import pandas as pd
import numpy as np
from numpy.random import RandomState, randint, uniform, binomial
from psychopy import visual, core, event
from psychopy.data import StairHandler
import tools
from tools import check_quit, wait_check_quit


def main(arglist):

    # Get the experiment paramters
    mode = arglist.pop(0)
    p = tools.Params("punch_%s" % mode)
    p.set_by_cmdline(arglist)

    # Open up the stimulus window
    win = tools.launch_window(p)

    # Set up the stimulus objects
    fix = visual.GratingStim(win, tex=None, mask="circle",
                             color=p.fix_color, size=p.fix_size)
    stims = dict(frame=Frame(win, p),
                 dots=Dots(win, p),
                 fix=fix)
    globals()[mode](p, win, stims)


def behav(p, win, stims):

    state = RandomState(abs(hash(p.subject)))
    choices = list(letters[:8])
    p.sched_id = state.permutation(choices)[p.run - 1]
    design_file = "design/behav_%s.csv" % p.sched_id
    d = pd.read_csv(design_file)
    n_trials = len(d)

    tools.WaitText(win, "hello world", height=.7)(check_keys=["space"])

    stim_event = EventEngine(win, stims, p)

    with tools.PresentationLoop(win):
        for trial in xrange(n_trials):

            context = d.context[trial]
            stims["frame"].set_context(context)

            if d.early[trial]:
                early = True
                cue_dur = uniform(*p.cue_dur)
            else:
                early = False
                cue_dur = None

            motion = d.motion[trial]
            color = d.color[trial]
            target = [color, motion][context]
            res = stim_event(context, motion, color, target, early, cue_dur)

            stims["fix"].draw()
            win.flip()
            wait_check_quit(uniform(*p.iti))


def train(p, win, stims):

    tools.WaitText(win, "hello world", height=.7)(check_keys=["space"])

    stim_event = EventEngine(win, stims, p, feedback=True)

    trained = False
    with tools.PresentationLoop(win):
        while not trained:

            context = randint(2)
            stims["frame"].set_context(context)

            for trial in xrange(p.n_per_block):

                color = np.random.randint(len(p.dot_color_names))
                direction = np.random.randint(len(p.dot_dirs))
                target = color if context == "color" else direction
                res = stim_event(color, direction, context, target)

                stims["fix"].draw()
                stims["frame"].draw()
                win.flip()
                wait_check_quit(uniform(*p.iti))


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
        self.debug = p.debug
        if self.debug:
            self.debug_text = [visual.TextStim(win,
                                               pos=(0.0, -5.0),
                                               height=0.5),
                               visual.TextStim(win,
                                               pos=(0.0, 5.0),
                                               height=0.5)]

    def __call__(self, context, direction, color, target,
                 early=False, cue_dur=0):

        self.dots.new_positions()

        # Debugging information
        if self.debug:
            color_name = self.p.dot_color_names[color]
            direction_deg = self.p.dot_dirs[direction]
            msg1 = "Orient: %s   Color: %s" % (color_name, direction_deg)
            self.debug_text[0].setText(msg1)
            msg2 = ["motion", "color"][context]
            self.debug_text[1].setText(msg2)

        self.dots.new_colors(color)
        self.dots.new_directions(direction)

        # Early Cue Presentation
        if early:
            self.frame.draw()
            self.fix.draw()
            self.win.flip()
            wait_check_quit(cue_dur)

        # Main Stimulus Presentation
        resp_clock = core.Clock()
        event.clearEvents()
        for frame in xrange(self.p.stim_flips):
            self.fix.draw()
            self.dots.draw()
            self.frame.draw()
            if self.debug:
                for text in self.debug_text:
                    text.draw()
            self.win.flip()

        # Response Collection
        correct = False
        rt = np.nan
        keys = event.getKeys(timeStamped=resp_clock)
        for key, stamp in keys:
            if key in ["q", "escape"]:
                print "Subject quit execution"
                core.quit()
            elif key in self.resp_keys:
                key_idx = self.resp_keys.index(key)
                rt = stamp
                if key_idx == target:
                    correct = True

        # Feedback
        if self.feedback and not correct:
            for frame in xrange(60):
            #for frame in xrange(self.p.fb_frames):
                if not frame % 10:
                #if not frame % self.p.fb_freq:
                    self.frame.flip_phase()
                self.frame.draw()
                self.fix.draw()
                self.win.flip()

            self.frame.reset_phase()

        return None


class Frame(object):

    def __init__(self, win, p):

        self.lr = []
        self.tb = []
        self.field_size = p.field_size
        self.frame_width = p.frame_width
        self.sf_list = p.frame_sfs
        self.base_phase = 0 if p.window_color in ["black", -1] else 0.5

        for pos in [-.5, .5]:
            self.lr.append(visual.GratingStim(win, tex=p.frame_tex,
                                              contrast=p.frame_contrast,
                                              phase=self.base_phase,
                                              pos=(0, pos * p.field_size)))

            self.tb.append(visual.GratingStim(win, tex=p.frame_tex,
                                              contrast=p.frame_contrast,
                                              phase=self.base_phase,
                                              pos=(pos * p.field_size, 0)))

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

    def flip_phase(self):

        for stims in [self.tb, self.lr]:
            for i in range(2):
                stims[i].setPhase((stims[i].phase + .5) % 1)

    def reset_phase(self):

        for stims in [self.tb, self.lr]:
            for i in range(2):
                stims[i].setPhase(self.base_phase)

    def draw(self):

        for stims in [self.tb, self.lr]:
            for i in range(2):
                stims[i].draw()


class Dots(object):

    def __init__(self, win, p):

        self.speed = p.dot_speed / 60
        self.colors = p.dot_color_names
        self.dot_base_hue = p.dot_base_hue
        self.dot_sat = p.dot_sat
        self.dot_val = p.dot_val
        self.dirs = p.dot_dirs
        self.field_size = p.field_size
        self.ndots = p.dot_number
        self.dot_life_mean = p.dot_life_mean
        self.dot_life_std = p.dot_life_std
        self.dimension = len(p.dot_color_names)
        assert self.dimension == len(p.dot_dirs)

        dot_shape = None if p.dot_shape == "square" else p.dot_shape
        self.dots = visual.ElementArrayStim(win, "deg",
                                            fieldShape="square",
                                            fieldSize=p.field_size,
                                            nElements=p.dot_number,
                                            sizes=p.dot_size,
                                            elementMask=dot_shape,
                                            colors=np.ones((p.dot_number, 3)),
                                            elementTex=None,
                                            )

        self.new_signals(p.dot_col_coh, p.dot_mot_coh)

        self.dot_life = np.round(np.random.normal(p.dot_life_mean,
                                                  p.dot_life_std,
                                                  size=p.dot_number))

    def new_signals(self, col_coh, mot_coh):

        col_signal = np.zeros(self.ndots, bool)
        mot_signal = np.zeros(self.ndots, bool)

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
        t_hue = target_color / self.dimension
        t_hue = t_hue + self.dot_base_hue
        t_hue = t_hue - np.floor(t_hue)
        hues[self.col_signal] = t_hue
        s = self.dot_sat
        v = self.dot_val
        colors = np.array([colorsys.hsv_to_rgb(h, s, v) for h in hues])
        colors = colors * 2 - 1
        self.dots.setColors(colors)

    def new_directions(self, target_dir):

        dirs = np.random.uniform(size=self.ndots) * 2 * np.pi
        dirs[self.mot_signal] = np.deg2rad(self.dirs[target_dir])
        self._dot_dirs = dirs

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
        xys[:, 0] += self.speed * np.cos(self._dot_dirs)
        xys[:, 1] += self.speed * np.sin(self._dot_dirs)
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


if __name__ == "__main__":
    main(sys.argv[1:])
