from __future__ import division
import sys
import os.path as op
from textwrap import dedent
from string import letters
import pandas as pd
import numpy as np
from numpy.random import RandomState, randint, uniform, binomial
from psychopy import visual, core, event
from psychopy.data import StairHandler
import psychopy.monitors.calibTools as calib
import tools
from tools import check_quit, wait_check_quit


def main(arglist):

    # Get the experiment paramters
    mode = arglist.pop(0)
    p = tools.Params("punch_%s" % mode)
    p.set_by_cmdline(arglist)

    # Open up the stimulus window
    calib.monitorFolder = "./calib"
    mon = calib.Monitor(p.monitor_name)
    m = tools.WindowInfo(p, mon)
    win = visual.Window(**m.window_kwargs)

    # Set up the stimulus objects
    fix = visual.PatchStim(win, tex=None, mask="circle",
                           color=p.fix_color, size=p.fix_size)
    stims = dict(frame=Frame(win, p),
                 dots=Dots(win, p),
                 fix=fix)
    globals()[mode](p, win, stims)


def behav(p, win, stims):

    tools.WaitText(win, "hello world", height=.7)(check_keys=["space"])

    stim_event = EventEngine(win, stims, p)

    with tools.PresentationLoop(win):
        for i in xrange(12):

            rel_dim = randint(2)
            dim_color = p.frame_colors[rel_dim]
            stims["frame"].set_color(dim_color)

            if binomial(1, p.early_cue_prob):
                early = True
                cue_dur = uniform(*p.cue_dur)
            else:
                early = False
                cue_dur = None

            color = np.random.randint(len(p.dot_colors))
            direction = np.random.randint(len(p.dot_dirs))
            res = stim_event(color, direction, early, cue_dur)

            stims["fix"].draw()
            win.flip()
            wait_check_quit(uniform(*p.iti))


def train(p, win, stims):

    tools.WaitText(win, "hello world", height=.7)(check_keys=["space"])

    stim_event = EventEngine(win, stims, p)


class EventEngine(object):

    def __init__(self, win, stims, p, debug=False):

        self.win = win
        self.stims = stims
        self.frame = stims["frame"]
        self.dots = stims["dots"]
        if "fb" in stims:
            self.fb = stims["fb"]
        self.p = p
        self.stim_dur = p.stim_dur
        self.resp_keys = p.resp_keys
        self.debug = p.debug
        if self.debug:
            self.debug_text = visual.TextStim(win,
                                              pos=(0.0, -5.0),
                                              height=0.5)

    def __call__(self, color, direction, early, cue_dur, target=None):

        self.dots.new_positions()

        color_name = self.p.dot_color_names[color]
        direction_deg = self.p.dot_dirs[direction]

        if self.debug:
            msg = "Color: %s; Direction: %s" % (color_name, direction_deg)
            self.debug_text.setText(msg)

        self.dots.new_colors(color)
        self.dots.new_directions(direction)

        if early:
            self.frame.draw()
            self.win.flip()
            wait_check_quit(cue_dur)

        for frame in xrange(self.p.stim_flips):
            self.frame.draw()
            self.dots.draw()
            if self.debug:
                self.debug_text.draw()
            self.win.flip()
        check_quit()

        return None


class Frame(object):

    def __init__(self, win, p):

        self.fix = visual.PatchStim(win, tex=None, mask="circle",
                                    color=p.fix_color, size=p.fix_size)
        frame_size = p.dot_field_size + 2 * p.frame_width
        self.frame = visual.PatchStim(win, tex=None, mask=p.frame_shape,
                                      opacity=p.frame_opacity,
                                      size=frame_size)
        inner_size = p.dot_field_size + 2 * p.dot_size
        self.inner = visual.PatchStim(win, tex=None, color=p.window_color,
                                      mask=p.frame_shape,
                                      size=inner_size)

    def set_color(self, color):

        self.frame.setColor(color)

    def draw(self):

        self.frame.draw()
        self.inner.draw()
        self.fix.draw()


class Dots(object):

    def __init__(self, win, p):

        self.speed = p.dot_speed / 60
        self.colors = p.dot_colors
        self.dirs = p.dot_dirs
        self.field_size = p.dot_field_size
        self.ndots = p.dot_number
        self.dot_life_mean = p.dot_life_mean
        self.dot_life_std = p.dot_life_std
        self.dimension = len(p.dot_colors)
        assert self.dimension == len(p.dot_dirs)

        dot_shape = None if p.dot_shape == "square" else p.dot_shape
        self.dots = visual.ElementArrayStim(win, "deg",
                                            fieldShape="square",
                                            fieldSize=p.dot_field_size,
                                            nElements=p.dot_number,
                                            sizes=p.dot_size,
                                            elementMask=dot_shape,
                                            colors=np.ones((p.dot_number, 3)),
                                            elementTex=None,
                                            )

        # Build an index into the dot list for colors
        n_colors = len(p.dot_colors)
        per_cell = p.dot_number / n_colors
        extra_targets = per_cell * p.dot_col_coh
        n_target = per_cell + extra_targets
        n_distract = per_cell - extra_targets / (n_colors - 1)
        color_ids = [np.repeat(0, n_target)]
        color_ids += [np.repeat(i, n_distract) for i in range(1, n_colors)]
        color_ids = np.concatenate(color_ids)
        #assert len(color_ids) == p.dot_number
        self.color_ids = color_ids

        # Now figure out which dots should move coherently
        signal_dots = np.ones(color_ids.shape, bool)
        for i in range(n_colors):
            n_dots = np.sum(color_ids == i)
            n_signal = n_dots * p.dot_mot_coh
            n_noise = n_dots - n_signal
            signal_dots[color_ids == i] = np.concatenate(
                [np.repeat(True, n_signal), np.repeat(False, n_noise)])
        #assert signal_dots.sum() / p.dot_number == p.dot_mot_coh
        self.signal_dots = signal_dots

        self.dot_life = np.round(np.random.normal(p.dot_life_mean,
                                                  p.dot_life_std,
                                                  size=p.dot_number))

    def new_colors(self, target_color):

        colors = np.zeros((self.ndots, 3))
        colors[self.color_ids == 0] = self.colors[target_color]
        
        j = 1  # We need two indices
        for i, color in enumerate(self.colors):
            if i != target_color:
                colors[self.color_ids == j] = color
                j += 1

        self.dots.setColors(colors)

    def new_directions(self, target_dir):

        dirs = np.random.rand(self.ndots) * 2 * np.pi
        dirs[self.signal_dots] = np.deg2rad(self.dirs[target_dir])
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
