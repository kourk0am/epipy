import numpy as np

class SEIRC(object):
    """ This class represents the SEIR epdidemiological model with added clinical estimates. 
    See https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model for the classical SEIR model.
    
    tldr: self.par; self.simulate(); self.result
    
    The 'removed' population from the clasical SEIR is split between mild and severe cases which recover at different 
    timescales. All mild cases recover. Some severe cases become fatal. Fraction of severe cases which become fatal depends on healthcare 
    system load. If threre are more severe cases than the healthcare can handle then the 'overflow'severe cases, 
    which cannot fit into the healthcare capacity, have increased fatality rate.
    
    Parameters for the simulation are given in self.par dictionary. This can be passed to the SEIRC object upon creation or changed later. 
    The dictionary looks like this:
    if self.par == None:
            self.par = {'N': 70e6, # total population
                        'I0' : 1, # number of initial infected individuals
                        'R' : 3.5, #reproduction number
                        'T_inc' : 5.2, # incubation time - no symptoms, not infectious
                        'T_inf' : 2.9, # infectious time - no symptoms, infectious
                        'T_mild' : 14, # recovery time for mild cases
                        'T_severe' : 30, # recovery time for severe cases
                        'p_mild' : 0.8, # probability of a mild case
                        'p_fatal' : 0.02, # probability of a fatal case
                        'p_fatal_overrun' : 0.05, # probability of a fatal case in an overrun healthcare system
                        'healthcare_capacity' : 30e3, # maximum number of severe cases that the healthcare system can handle
                        'intervention' : True, # do we do an intervention
                        'R_intervention' : [0.7,1.3], # R after intervention
                        'intervention_intervals' : [[50,300], [300,600]], # during which time intervals we do the intervention [start, stop]
                        'start_day' : 0, # first day of the outbreak - change this to shift time axis of the simulation
                        'duration' : 600 # duration of the simulation
                       }
    
    p_mild is probability that a random individual ends up in the mild category if infected (probability for ending up in the severe category
    is calculated automatically as 1-p_mild)
    
    p_fatal is probability that a random individual dies if infected and if given all available help from healthcare. Probability that 
    an individual in severe category dies is calculated from this and p_mild and load on the
    healthcare system. Make sure that p_fatal<1-p_mild
    
    p_fatal_overrun is probability that a random individual dies if infected and not given all available help from healthcare 
    (too many patients, go home, good luck). Probability that an individual in severe category dies is calculated from this, p_mild and load on the
    healthcare system. Make sure that p_fatal_overrun<1-p_mild
    
    healthcare_capacity is the maximu number of severe cases that the healthcare system can handle at one time. Any extra severe cases end up 
    in the overrun category and have an increased probability of dying. 
    
    intervention - if set to True, R is going to be changed according to R_intervention and intervention_intervals. If it is false R stays constant.
    
    'R_intervention' is a list of R values during intervention intervals.
    
    intervention_intervals is a list of [start,stop] intervals during which the corresponding R values are applied (first R during first interval, second R during the second...). 
    Outside of the intervals R goes back to its original value.
    R_intervention needs to have the same number of items as intervention_intervals. The intervals should not overlap.             
    
    Simulation is run with self.simulate(dt = 0.01). If any of the characteristic times is less than 1 consider using shorter dt than the default. 
    Results are written in self.result - a dict with arrays for time, susceptible, exposed, infected, mild, severe, dead and recovered populations. 
    Removed population from the clasical SEIR and the R values are given as well. The healthcare load is also given - this is ratio of (active severe cases)/(healthcare capacity)
    
    """
    def __init__(self, parameters = None):
        self.par = parameters
        
        # check if we got parameter dictionary, if not, create one
        # we should also check whether the par dict is valid but this can be added later...
        if self.par == None:
            self.par = {'N': 70e6, # total population
                        'I0' : 1, # number of initial infected individuals
                        'R' : 3.5, #reproduction number
                        'T_inc' : 5.2, # incubation time - no symptoms, not infectious
                        'T_inf' : 2.9, # infectious time - no symptoms, infectious
                        'T_mild' : 14, # recovery time for mild cases
                        'T_severe' : 30, # recovery time for severe cases
                        'p_mild' : 0.8, # probability of a mild case
                        'p_fatal' : 0.02, # probability of a fatal case
                        'p_fatal_overrun' : 0.05, # probability of a fatal case in an overrun healthcare system
                        'healthcare_capacity' : 30e3, # maximum number of severe cases that the healthcare system can handle
                        'intervention' : True, # do we do an intervention
                        'R_intervention' : [0.7,1.3], # R after intervention
                        'intervention_intervals' : [[50,300], [300,600]], # during which time intervals we do the intervention [start, stop]
                        'start_day' : 0, # first day of the outbreak - change this to shift time axis of the simulation
                        'duration' : 600 # duration of the simulation
                       }
        # calculate p_severe 
        self.par['p_severe'] = 1 - self.par['p_mild'] 
        
    
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
        
    def simulate(self,  dt = 0.01):
        """Runs a simulation with parameters given in self.par. Results are stored in self.result."""
        duration = self.par['duration']
        steps = int(duration/dt)
        
        t = np.zeros(steps)    # time
        t[0] = self.par['start_day']
        su = np.zeros(steps)   # susceptible population
        su[0] = self.par['N']  # initial susceptible population
        e = np.zeros(steps)    # exposed population
        i = np.zeros(steps)    # infectious population
        i[0] = self.par['I0']  #initial infectious population
        m = np.zeros(steps)    # population in mild state
        s = np.zeros(steps)    # population in severe state within healthcare
        r = np.zeros(steps)    # recovered
        d = np.zeros(steps)    # dead
        rm = np.zeros(steps)   # removed - people who show symptoms and are excluded from population: sum of ill(in any state) recovered and dead
        rv = np.zeros(steps)   # R values
        
        N = self.par['N']
        R0 = self.par['R']
        T_inc = self.par['T_inc']
        T_inf = self.par['T_inf']
        T_m = self.par['T_mild']
        T_s = self.par['T_severe']
        p_m = self.par['p_mild']
        p_s = self.par['p_severe']
        p_f = self.par['p_fatal']/self.par['p_severe']          # probability that a severe case turns fatal, mild cases do not turn fatal
        p_fo = self.par['p_fatal_overrun']/self.par['p_severe'] # probability that a severe case outside healthcare system turns fatal
        
        #p_so = self.par['p_severe_overrun']
        hc =self.par['healthcare_capacity']
        intervention = self.par['intervention']
        R1 = self.par['R_intervention']
        
        
        R = R0
        #print('p_f = ' + str(p_f))
        #print('p_fo = ' + str(p_fo))
        for n in range(steps-1):
            t[n+1] = t[n] + dt
            
            # calculate effective probability that severe case is fatal
            p_fe = p_f 
            if s[n] > hc: #if healthcare system is overrun
                p_fe = (p_f*hc + p_fo*(s[n]-hc))/s[n] # weighted average of fatality rates inside and outside healthcare system
            
            #if t[n]%10 < 0.05:
            #    print p_fe
            
            if intervention:
                R = self.getR(t[n])
                
            # calculate changes in individual populations    
            dsu = -R/(T_inf*N)*su[n]*i[n]*dt              # change of susceptible population 
            de = (R*su[n]*i[n]/(T_inf*N) - e[n]/T_inc)*dt # change of exposed population
            di = (e[n]/T_inc - i[n]/T_inf)*dt             # change of infectious population
            drm = i[n]/T_inf*dt                           # change of removed population
            dm = (i[n]/T_inf*p_m - m[n]/T_m)*dt           # change of mild population
            ds = (i[n]/T_inf*p_s - s[n]/T_s)*dt           # change of severe population
            dr = (m[n]/T_m + (1 - p_fe)*s[n]/T_s)*dt      # change of recovered population
            dd = (p_fe*s[n]/T_s)*dt                       # change of dead population
            
            # propagate to the next step
            su[n+1] = su[n] + dsu
            e[n+1] = e[n] + de
            i[n+1] = i[n] + di
            rm[n+1] = rm[n] + drm
            m[n+1] = m[n] + dm
            s[n+1] = s[n] + ds
            r[n+1] = r[n] + dr
            d[n+1] = d[n] + dd
            rv[n] = R
        
        rv[-1] = R
            
        out = {"time":t,
              "susceptible": su,
              'exposed':e,
              'infectious':i,
              'mild':m,
              'severe':s,
              'recovered':r,
              'dead' : d,
              'removed':rm,
              'healthcare load': s/hc,
              'R' : rv}
        
        self.result = out
        