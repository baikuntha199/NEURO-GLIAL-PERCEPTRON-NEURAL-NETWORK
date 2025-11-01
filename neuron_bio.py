import numpy as np
from .synapses import nmda_mag_block

# --------- single compartment with HH + Ca + K_Ca ---------
class Compartment:
    def __init__(self, dt=0.025, C=1.0,
                 gNa=120.0, gK=36.0, gL=0.3,
                 ENa=50.0, EK=-77.0, EL=-54.4,
                 gCa=0.5, ECa=120.0, gKCa=2.0,
                 Ca_rest=0.05, tau_Ca=80.0, alpha_Ca=0.002, Kd_KCa=0.3,
                 E_AMPA=0.0, E_NMDA=0.0, E_GABAA=-70.0, E_GABAB=-95.0):
        self.dt=dt; self.C=C
        self.gNa=gNa; self.gK=gK; self.gL=gL
        self.ENa=ENa; self.EK=EK; self.EL=EL
        self.gCa=gCa; self.ECa=ECa; self.gKCa=gKCa
        self.Ca_rest=Ca_rest; self.tau_Ca=tau_Ca; self.alpha_Ca=alpha_Ca; self.Kd_KCa=Kd_KCa
        self.E_AMPA=E_AMPA; self.E_NMDA=E_NMDA; self.E_GABAA=E_GABAA; self.E_GABAB=E_GABAB

        self.V = -65.0
        self.m,self.h,self.n = self._init_gates(self.V)
        self.mCa = 0.0
        self.Ca_i = Ca_rest

        # syn conductances set externally each step
        self.gA = 0.0; self.gN = 0.0; self.gGA = 0.0; self.gGB = 0.0

    # HH kinetics
    @staticmethod
    def alpha_m(V): return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0)/10.0))
    @staticmethod
    def beta_m(V):  return 4.0*np.exp(-(V+65.0)/18.0)
    @staticmethod
    def alpha_h(V): return 0.07*np.exp(-(V+65.0)/20.0)
    @staticmethod
    def beta_h(V):  return 1.0/(1.0 + np.exp(-(V+35.0)/10.0))
    @staticmethod
    def alpha_n(V): return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0)/10.0))
    @staticmethod
    def beta_n(V):  return 0.125*np.exp(-(V+65.0)/80.0)
    @staticmethod
    def mCa_inf(V): return 1.0/(1.0 + np.exp(-(V+20.0)/6.0))
    @staticmethod
    def tau_mCa(V): return 5.0 + 20.0/(1.0 + np.exp((V+25.0)/5.0))

    def _init_gates(self, V):
        am,bm=self.alpha_m(V),self.beta_m(V)
        ah,bh=self.alpha_h(V),self.beta_h(V)
        an,bn=self.alpha_n(V),self.beta_n(V)
        m=am/(am+bm); h=ah/(ah+bh); n=an/(an+bn)
        return m,h,n

    def set_syn(self, gA, gN, gGA, gGB):
        self.gA, self.gN, self.gGA, self.gGB = gA, gN, gGA, gGB

    def _currents(self, V):
        gNa_eff = self.gNa * (self.m**3) * self.h
        gK_eff  = self.gK  * (self.n**4)
        INa = gNa_eff * (V - self.ENa)
        IK  = gK_eff  * (V - self.EK)
        IL  = self.gL  * (V - self.EL)

        # Ca
        mCa_inf = self.mCa_inf(V); tau = self.tau_mCa(V)
        self.mCa += self.dt * (mCa_inf - self.mCa) / tau
        ICa = self.gCa * (self.mCa**2) * (V - self.ECa)

        # K_Ca (Hill on Ca)
        mKCa = self.Ca_i / (self.Ca_i + self.Kd_KCa + 1e-9)
        IKCa = self.gKCa * mKCa * (V - self.EK)

        # Synapses (with NMDA Mg block)
        B = nmda_mag_block(V, Mg_mM=1.0)
        IsA = self.gA * (V - self.E_AMPA)
        IsN = (self.gN * B) * (V - self.E_NMDA)
        IsGA= self.gGA * (V - self.E_GABAA)
        IsGB= self.gGB * (V - self.E_GABAB)
        return INa, IK, IL, ICa, IKCa, IsA, IsN, IsGA, IsGB

    def step(self, Iext=0.0, V_couple=0.0, g_axial=0.0):
        V = self.V; dt=self.dt
        # update HH gates
        am,bm=self.alpha_m(V),self.beta_m(V)
        ah,bh=self.alpha_h(V),self.beta_h(V)
        an,bn=self.alpha_n(V),self.beta_n(V)
        self.m += dt*(am*(1-self.m)-bm*self.m)
        self.h += dt*(ah*(1-self.h)-bh*self.h)
        self.n += dt*(an*(1-self.n)-bn*self.n)

        INa,IK,IL,ICa,IKCa,IsA,IsN,IsGA,IsGB = self._currents(V)
        I_syn = IsA + IsN + IsGA + IsGB
        I_ax  = g_axial * (V - V_couple)
        I_tot = INa + IK + IL + ICa + IKCa + I_syn + I_ax

        dV = (-I_tot + Iext) / self.C
        self.V = V + dt * dV

        # Ca dynamics: inward ICa only
        Ca_influx = -ICa * (ICa < 0.0)
        self.Ca_i += dt * ( self.alpha_Ca * Ca_influx - (self.Ca_i - self.Ca_rest)/self.tau_Ca )
        self.Ca_i = max(0.0, self.Ca_i)
        return self.V

# --------- two-compartment neuron (dendrite + soma) ---------
class BioNeuron:
    def __init__(self, dt=0.025, geom=None, **chan):
        self.dt = dt
        self.g_axial = (geom or {}).get('g_axial_mScm2', 0.5)
        self.soma = Compartment(dt=dt, **chan)
        self.dend = Compartment(dt=dt, **chan)

    def set_syn_dend(self, gA, gN, gGA, gGB):
        self.dend.set_syn(gA, gN, gGA, gGB)

    def set_syn_soma(self, gA, gN, gGA, gGB):
        self.soma.set_syn(gA, gN, gGA, gGB)

    def step(self, Iext_soma=0.0):
        Vs_prev, Vd_prev = self.soma.V, self.dend.V
        Vd = self.dend.step(Iext=0.0, V_couple=Vs_prev, g_axial=self.g_axial)
        Vs = self.soma.step(Iext=Iext_soma, V_couple=Vd_prev, g_axial=self.g_axial)
        return Vs, Vd
