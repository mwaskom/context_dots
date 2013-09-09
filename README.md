# Experimental Code for PUnCH Project

## Project Details 

The PUnCH (or Parametric Uncertainty in the Control Hierarchy) experiment investigates the behavioral and neural effects of parametrically manipulating uncertainty at multiple levels of task structure. 

The basic task used here is a context-dependent random dot motion perceptual
decision-making task.

## Code Structure

The code below uses [PsychoPy](http://www.psychopy.org/) for the presentation
of experimental stimuli. It also makes heavy use of the
[cregg](https://github.com/mwaskom/cregg) package for experimental utilities.
Design generation uses the [moss](https://github.com/mwaskom/moss) package.
Otherwise the standard SciPy stack should be sufficient to control the
experiment.


### punch.py

This script exposes the main interface for collecting data. There are several
sub-modes of this script for presenting different stages of the experiment
(these are detailed below).

All modes take a `-subject` (can be shortened to `-s`) argument. Several
aspects are randomized across subjects based on the subject ID, so any
interaction with a subject should use this flag. The `-cbid` option can be
used to seed the counterbalance with a different ID than will be associated
with the data.

Run with the `-debug` flag to present in a separate GUI window. Debug mode
also adds textual information about the stimulus parameters on the screen.

The `punch.py` interface has the following modes (usage is `python punch.py <mode>`):

#### instruct

Self-paced participant-specific instructions for the experiment.

#### learn

Full-coherence stimuli, blocked, with feedback. Terminates when participant
reaches a set criterion of performance (in terms of accuracy).

#### staircase

Blocked trials that staircase the coherence values (independently for each
dimension) based on response accuracy. Generates a json file in the data
directory to be used for subsequent sessions. Presents response feedback.

#### practice

Balanced, interleaved context-switching version of the experiment. The stimulus
values and context choices are generated at runtime, so this might not be
perfectly balanced for short runs.

Feedback can be turned on (it is off by default) with the `-feedback` command 
line switch. The number of trials in the run and the number of trials in between
breaks can also be set on the command line. Can also run in scanner mode.

#### scan

The main experiment. Reads trial schedule information produced by
`generate_designs.py`. Can be collected in scanner mode or computer mode,
depending on usage of the `-fmri` switch.

### params.py

File with dictionaries containing experimental parameters.

### generate_designs.py

Module for creating static schedules for the different modes of the experiment.
At the moment it is only used for the `scan` mode; everything else is generated
at runtime.

### **monitors.py**

File with monitor information.

### design/

Static design files from `generate_designs.py` execution are stored here.

### calib/

Contains monitor calibration information.

### data/

Experimental data are generated here
