

Model-S -- Symbiosis Promotes Fitness Improvements in the Game of Life
======================================================================

Peter Turney
July 22, 2021

Model-S is a tool for modeling symbiosis. Model-S is implemented as a 
set of Python scripts that work with the Golly software for the Game 
of Life.

This document describes how to install and run Model-S in Windows 10.
With some changes, you should also be able to run Model-S in Linux
or Mac OS.


Installing Model-S
==================

(1) Download and Install Golly

Golly is a C++ program for the simulation of cellular automata:

- https://en.wikipedia.org/wiki/Golly_(program)

I used the 64-bit Golly 4.0 for Windows 10 (golly-4.0-win-64bit.zip):

- http://golly.sourceforge.net/
- https://sourceforge.net/projects/golly/files/latest/download

Golly is stored in a zip file. I created a directory called Golly
and put the contents of the zip file in this directory:

- C:\Users\peter\Golly


(2) Download and Install Python

Golly can be extended with scripts written in Python or Lua. Model-S is
a set of Python scripts that run with Golly. 

I used Python 3.9.1 for Windows (python-3.9.1-amd64.exe). Golly 4.0
is designed to work with Python 3.3+. It no longer works with Python 2.X.

Here is some information on using Python with Golly:

- http://golly.sourceforge.net/Help/python.html


(3) Download and Install Numpy and Statistics

Numpy provides Python numerical functions needed by Model-S. After Python
has been installed, Numpy can be installed in Windows 10 by opening a
command prompt and executing the following commands:

> pip3 install numpy
> pip3 install wheel
> pip3 install statistics

You should be able to find pip3 here:

C:\Users\MyUserName\AppData\Local\Programs\Python\Python39\Scripts


(4) Download and Install Model-S

Create a subdirectory of your Golly directory and put the Model-S files
in this subdirectory. In my case, the Model-S files are in this
directory:

- C:\Users\peter\Golly\Model-S

Golly needs to know where to find the rules for the Immigration game.
The rules are in the file Immigration.rule in the Model-S files.
Start Golly and navigate through the Golly menu system as follows:

- File > Preferences > Control > Your Rules ...

Click on "Your Rules ..." and enter the Model-S directory:

- C:\Users\peter\Golly\Model-S


(5) Adjust Windows 10 Antimalware Service

Windows 10 Antimalware Service wastes a lot of CPU time checking Golly
for malware, whenever Golly is executing. To free up your CPU, tell the
Antimalware Service not to check Golly:

- open Windows Defender Security Center
- select Virus & threat protection
- select Virus & threat protection settings
- select Add or remove exclusions
- add the Golly process (Golly.exe)


(6) Adjust Windows 10 Update Policy

Windows 10 will periodically install updates and restart the computer
without asking for permission. This will terminate an ongoing simulation
prematurely. To prevent this, you need to use the Windows Group Policy
Editor, which is available in Windows 10 Pro, but not in Windows 10 Home.
If you have Windows 10 Home, it is worthwhile to upgrade to Windows 10 Pro.
Here is information about how to set the Windows Group Policy Editor
to prevent termination of a simulation run:

- https://docs.microsoft.com/en-us/windows/deployment/update/waas-restart

I set my group policy as follows:

- Local Computer Policy > Computer Configuration > Administrative Templates
  > Windows Components > Windows Update
- Configure Automatic Updates = Enabled = 2 = Notify before downloading
  and installing any updates
- No auto-restart with logged on users for scheduled automatic updates
  instalations = Enabled
  

Running Model-S
===============

(1) run_model.py -- run a simulation; evolve a population

The main routine for running Model-S is run_model.py. It uses the
supporting code in model_classes.py, model_functions.py, and
model_parameters.py. It also uses the rules for the Immigration
Game, in the file Immigration.rule.

To run Model-S, start Golly and then open the Model-S folder in the 
left panel of Golly. Click on run_model.py to start the simulation. 
You can control the behaviour of the simulation by editing the
numbers in the parameter file, model_parameters.py, before you
start Golly. 

When Golly is running, the Golly screen will show the final outcome of
each Immigration game that is played. The intermediate stages of the 
games are not displayed, in order to maximize the speed of the simulation. 
If you want a more detailed view of an individual game, use the
script view_contest.py. A typical simulation run takes about two to
six days for 100 generations, depending on the speed of the computer
and the settings of the parameters.

As run_model.py executes, it stores a log file with some statistics
from the run. It also stores samples (pickles) of individuals that 
evolve during the run. The directory where these files are stored, 
log_directory, is specified in model_parameters.py. You should
create a folder for storing the files and edit model_parameters.py
so that log_directory points to your desired folder.

(2) measure_areas.py -- calculate the average areas of individuals

After a simulation ends, measure_areas.py can examine samples to
calculate their areas (the number of rows times the number of columns).
In general, we expect the areas to grow over the generations.


(3) measure_densities.py -- calculate the average densities

After a simulation ends, measure_densities.py can examine samples to
calculate their densities (the number of 1s divided by the area).
We expect that a relatively narrow range of densities will be
preferred.


(4) measure_diversities.py -- calculate the standard deviation of fitness

After a simulation ends, measure_diversities.py can examine samples
to calculate the standard deviation of the fitness in the samples, which
gives an indication of how much diverity there is in the samples.

