# Experimental Code for PUnCH Project

## Project Details 

The PUnCH (or Parametric Uncertainty in the Control Hierarchy) experiment investigates the behavioral and neural effects of parametrically manipulating uncertainty at multiple levels of task structure. 

## Code Structure

The code below uses PsychoPy for the presentation of experimental stimuli.
It also makes heavy use of the cregg package for experimental utilities.
Design generation uses the moss package. Otherwise the standard SciPy stack
should be sufficient to control the experiment.


All files take a `-subject` (can be shortened to `-s`) argument. Several
aspects are randomized across subjects based on the subject ID, so any
interaction with a subject should use this flag.

### punch.py

This script exposes the main interface for collecting data. There are several
sub-modes of this script for presenting different stages of the experiment
(these are detailed below).

Run with the `-debug` flag to present in a separate GUI window. Debug mode
also adds textual information about the stimulus parameters on the screen.

#### instruct

Self-paced participant-specific instructions for the experiment.

#### learn

Full-coherence stimuli, blocked, with feedback. Terminates when participant
reaches a set criterion of performance (in terms of accuracy).

#### staircase

Blocked trials that staircase the coherence values (independently for each
dimension) based on response accuracy. Generates a json file in the data
directory to be used for subsequent sessions.

#### practice

Balanced, interleaved context-switching version of the experiment. The stimulus
values and context choices are generated at runtime, so this might not be
perfectly balanced for short runs.

#### scan

The main experiment. Reads trial schedule information produced by
`generate_designs.py`. Can be collected in scanner mode or computer mode,
depending on presence of `-fmri` switch.

### params.py

File with dictionaries containing experimental parameters.

### generate_designs.py

Module for creating static schedules for the different modes of the experiment.
At the moment it is only response for the `scan` mode; everything else is
generated at runtime.

### **monitors.py**

File with monitor information.

### design/

Static design files from `generate_designs.py` execution are stored here.

### calib/

Contains monitor calibration information.

### data/

Experimental data are generated here
