"""Hold information about different monitors."""
from textwrap import dedent

cni_30 = dict(monitor_name='cni_30',
              calib_file='calib/cni_lums_20110718.csv',
              calib_date='20110718',
              width=64.3,
              distance=205.4,
              size=[1280, 800],
              notes=dedent("""
              30" Flat panel display
              Parameters taken from the CNI wiki:
              http://cni.stanford.edu/wiki/MR_Hardware#Flat_Panel.
              Accessed on 8/9/2011.
              """))

cni_47 = dict(monitor_name='cni_47',
              width=103.8,
              distance=277.1,
              size=[1920, 1080],
              notes=dedent('47" 3D LCD display - not yet calibrated'))

mlw_mbpro = dict(monitor_name='mlw-mbpro',
                 width=33.2,
                 size=[1440, 900],
                 distance=63,
                 notes="")

mlw_mbair = dict(monitor_name='mlw-mbair',
                 width=30.5,
                 size=[1440, 900],
                 distance=63,
                 notes="")

ben_octocore = dict(monitor_name='ben-octocore',
                    width=43.5,
                    size=[1680, 1050],
                    distance=60,
                    notes=dedent("""Horizontal monitor on the Mac Pro
                                 in MLW's office."""))
