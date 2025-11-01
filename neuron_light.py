import numpy as np

class LIFNeuron:
    def __init__(self, dt=0.1, tau_m=20.0, EL=-65.0, Vth=-50.0, Vreset=-65.0, Rm=10.0):
        self.dt=dt; self.tau=tau_m; self.EL=EL; self.Vth=Vth; self.Vreset=Vreset; self.Rm=Rm
        self.V = EL

    def step(self, I_syn=0.0, I_ext=0.0):
        dV = (-(self.V - self.EL) + self.Rm*(I_syn + I_ext)) / self.tau
        self.V += self.dt * dV
        spk = self.V >= self.Vth
        if spk: self.V = self.Vreset
        return self.V, bool(spk)

class AdExNeuron:
    # Minimal adaptive exponential IF (for meso scale)
    def __init__(self, dt=0.1, C=200.0, gL=10.0, EL=-70.0, VT=-50.0, DeltaT=2.0,
                 a=2.0, tau_w=200.0, b=40.0, Vreset=-58.0):
        self.dt=dt; self.C=C; self.gL=gL; self.EL=EL; self.VT=VT; self.DeltaT=DeltaT
        self.a=a; self.tau_w=tau_w; self.b=b; self.Vreset=Vreset
        self.V = EL; self.w = 0.0

    def step(self, I_syn=0.0, I_ext=0.0):
        dV = ( -self.gL*(self.V-self.EL) + self.gL*self.DeltaT*np.exp((self.V-self.VT)/self.DeltaT) - self.w + (I_syn+I_ext) ) / self.C
        dw = ( self.a*(self.V-self.EL) - self.w ) / self.tau_w
        self.V += self.dt * dV
        self.w += self.dt * dw
        spk = self.V > 0.0
        if spk:
            self.V = self.Vreset
            self.w += self.b
        return self.V, bool(spk)
