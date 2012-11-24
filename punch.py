from __future__ import division
import sys
import os.path as op
from textwrap import dedent
from string import letters
import pandas as pd
import numpy as np
from numpy.random import RandomState, randint, uniform, binomial
from psychopy import visual, core, event
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
                 grate=Grating(win, p),
                 fix=fix)
    if mode == "train":
        stims["fb"] = Feedback(win, p)
    globals()[mode](p, win, stims)


def behav(p, win, stims):

    tools.WaitText(win, "hello world", height=.7)(check_keys=["space"])

    stim_event = EventEngine(win, stims, p.stim_dur, p.resp_keys)

    with tools.PresentationLoop(win):
        for i in xrange(4):

            rel_dim = randint(3)
            dim_color = p.frame_colors[rel_dim + randint(1)]
            stims["frame"].set_color(dim_color)

            if binomial(1, p.early_cue_prob):
                early = True
                cue_dur = uniform(*p.cue_dur)
            else:
                early = False
                cue_dur = None

            res = stim_event(early, cue_dur)

            stims["fix"].draw()
            win.flip()
            wait_check_quit(uniform(*p.iti))


def train(p, win, stims):

    tools.WaitText(win, "hello world", height=.7)(check_keys=["space"])

    stim_event = EventEngine(win, stims, p.stim_dur, p.resp_keys,
                             True, p.fb_colors, p.fb_dur)

    dim_colors = np.reshape(p.frame_colors, (3, 2))

    sample = lambda : tools.categorical([1 / 3, 1 / 3, 1 / 3])

    with tools.PresentationLoop(win):

        trained = False
        n_blocks = 0
        grate = stims["grate"]
        frame = stims["frame"]
        fix = stims["fix"]

        while not trained:
            dim = sample()
            dim_color = dim_colors[dim, tools.flip()]
            frame.set_color(dim_color)

            for trial in xrange(p.n_per_block):

                sf = sample()
                grate.set_sf(p.stim_sfs[sf])
                ori = sample()
                grate.set_ori(p.stim_oris[ori])
                contrast = sample()
                grate.set_contrast(p.stim_contrasts[contrast])

                correct = [sf, ori, contrast][dim]

                res = stim_event(False, None, correct)

                frame.draw()
                fix.draw()
                win.flip()
                wait_check_quit(uniform(*p.iti))

            n_blocks += 1
            if n_blocks > 2:
                trained = True


class EventEngine(object):

    def __init__(self, win, stims, stim_dur, resp_keys,
                 give_feedback=False, fb_colors=None, fb_dur=None):

        self.win = win
        self.stims = stims
        self.frame = stims["frame"]
        self.grate = stims["grate"]
        if "fb" in stims:
            self.fb = stims["fb"]
        self.stim_dur = stim_dur
        self.resp_keys = resp_keys
        self.give_feedback = give_feedback
        if give_feedback:
            self.fb_colors = fb_colors
            self.fb_dur = fb_dur

    def __call__(self, early, cue_dur, target=None):

        if early:
            self.frame.draw()
            self.win.flip()
            wait_check_quit(cue_dur)

        self.frame.draw()
        self.grate.draw()
        self.win.flip()
        wait_check_quit(self.stim_dur)

        if self.give_feedback:
            correct = tools.flip()
            self.fb.set_status(correct)
            self.frame.draw()
            self.fb.draw()
            self.grate.draw()
            self.win.flip()
            wait_check_quit(self.fb_dur)

        return None


class Frame(object):

    def __init__(self, win, p):

        self.fix = visual.PatchStim(win, tex=None, mask="circle",
                                    color=p.fix_color, size=p.fix_size)
        self.frame = visual.PatchStim(win, tex=None, mask=p.frame_shape,
                                      opacity=p.frame_opacity,
                                      size=p.frame_size)
        inner_size = p.frame_size - 2 * p.frame_width
        self.inner = visual.PatchStim(win, tex=None, color=0,
                                      mask=p.frame_shape,
                                      size=inner_size)

    def set_color(self, color):

        self.frame.setColor(color)

    def draw(self):

        self.frame.draw()
        self.inner.draw()
        self.fix.draw()


class Grating(object):

    def __init__(self, win, p):

        self.grate = visual.PatchStim(win, "sin", p.stim_mask,
                                      size=p.stim_size,
                                      opacity=p.stim_opacity)
        self.disk = visual.PatchStim(win, tex=None, mask=p.stim_mask,
                                     color=win.color,
                                     size=p.stim_disk_ratio)
        self.fix = visual.PatchStim(win, tex=None, mask="circle",
                                    color=p.fix_color, size=p.fix_size)

    def set_ori(self, ori):

        self.grate.setOri(ori)

    def set_contrast(self, contrast):

        self.grate.setContrast(contrast)

    def set_sf(self, sf):

        self.grate.setSF(sf)

    def draw(self):

        self.grate.draw()
        self.disk.draw()
        self.fix.draw()


class Feedback(object):

    def __init__(self, win, p):

        self.halo = visual.PatchStim(win, tex=None,
                                     mask="gauss",
                                     size=p.fb_size)
        self.fb_colors = p.fb_colors

    def set_status(self, correct):

        self.halo.setColor(self.fb_colors[correct])

    def draw(self):

        self.halo.draw()


if __name__ == "__main__":
    main(sys.argv[1:])
