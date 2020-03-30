import numpy as np

class SIR(object):
    """ This class represents the SIR epdidemiological model. It alows R0 to be changed during the course of simulation in order to mimick effect of measures taken to stop the spread of the disease.
    See https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model
    
    tldr: self.par; self.simulate(); self.result
    
    Parameters for the simulation are given in self.par dictionary. This can be passed to the SIR object upon creation or changed later. The dictionary looks like this:
     self.par = {'N': 70e6, # total population
                        'I0' : 1, # number of initial infected individuals
                        'R' : 3.5, #reproduction number
                        'T_inf' : 2.9, # infectious time (days)
                        'intervention' : True, # do we do an intervention
                        'R_intervention' : [0.7, 1.3], # R during intervention intervals
                        'intervention_intervals' : [[50,300], [300,600]], # during which time intervals we do the intervention [start, stop]
                        'start_day' : 0, # first day of the outbreak - change this to shift time axis of the simulation
                        'duration' : 600 # duration of the simulation
                       }
    
    There are no sanity checks for any of the parameters! 
                       
    intervention - if set to True, R is going to be changed according to R_intervention and intervention_intervals. If it is false R stays constant
    'R_intervention' is a list of R values during intervention intervals.
    intervention_intervals is a list of [start,stop] intervals during which the corresponding R values are applied (first R during first interval, second R during the second...). 
    Outside of the intervals R goes back to its original value.
    R_intervention needs to have the same number of items as intervention_intervals. The intervals should not overlap.                  
                      
    
    Simulation is run with self.simulate(dt = 0.01). If T_inf is less than 1 consider using shorter dt than the default. 
    Results are written in self.result - a dict with arrays for time, susceptible population, infected population, removed population and the R value
    
    """
    def __init__(self, parameters = None):
        self.par = parameters
        
        # check if we got parameter dictionary, if not, create one
        # we should also check whether the par dict is valid but this can be added later...
        if self.par == None:
            self.par = {'N': 70e6, # total population
                        'I0' : 1, # number of initial infected individuals
                        'R' : 3.5, #reproduction number
                        'T_inf' : 2.9, # infectious time
                        'intervention' : True, # do we do an intervention
                        'R_intervention' : [0.7, 1.3], # R during intervention intervals
                        'intervention_intervals' : [[50,300], [300,600]], # during which time intervals we do the intervention [start, stop]
                        'start_day' : 0, # first day of the outbreak - change this to shift time axis of the simulation
                        'duration' : 600 # duration of the simulation
                       }
       
        
    
    def isInInterval(self, x, interval):
        """Interval is expected to look like this [start, stop]. Returns True if start<x<stop and False otherwise."""
        value = False
        if x > interval[0] and x < interval[1]:
            value = True
        return value
    
    def isInIntervals(self, x, intervals):
        """Returns true if x is in any interval contained in intervals."""
        value = False
        for interval in intervals:
            if self.isInInterval(x, interval):
                value = True
        return value
    
    def getR(self, x):
        """Returns value of R for given time x."""
        R = self.par['R']
        for R_int, interval in zip (self.par['R_intervention'], self.par['intervention_intervals']):
            if self.isInInterval(x, interval):
                R = R_int
        return R
        
    def simulate(self, dt = 0.01):
        """Runs a simulation with parameters given in self.par. Results are stored in self.result."""
        duration = self.par['duration']
        steps = int(duration/dt)
        
        t = np.zeros(steps)           # time
        t[0] = self.par['start_day']  # first day of the outbreak
        su = np.zeros(steps)          # susceptible population
        su[0] = self.par['N']         # initial susceptible population
        i = np.zeros(steps)           # infectious population
        i[0] = self.par['I0']         # initial infectious population
        rm = np.zeros(steps)          # removed 
        rv = np.zeros(steps)          # R values - for reference
        
        
        N = self.par['N']
        R = self.par['R']
        T_inf = self.par['T_inf']
        intervention = self.par['intervention']
        
        for n in range(steps-1):
            t[n+1] = t[n] + dt
            
            if intervention:
                R = self.getR(t[n])
            
                
            # calculate changes in individual populations    
            dsu = -R/(T_inf*N)*su[n]*i[n]*dt              # change of susceptible population 
            di = (R/(T_inf*N)*su[n]*i[n] - i[n]/T_inf)*dt             # change of infectious population
            drm = i[n]/T_inf*dt                           # change of removed population
            
            
            # propagate to the next step
            su[n+1] = su[n] + dsu
            i[n+1] = i[n] + di
            rm[n+1] = rm[n] + drm
            rv[n] = R 
            
        rv[-1] = R    
        out = {"time":t,
              "susceptible": su,
              'infectious':i,
              'removed':rm,
               'R':rv
              }
        
        self.result = out
        