import numpy as np

# ---------- helpers ----------
def nmda_mag_block(V_mV: float | np.ndarray, Mg_mM: float = 1.0):
    # Jahr & Stevens style block
    return 1.0 / (1.0 + Mg_mM/3.57 * np.exp(-0.062 * V_mV))

# ---------- STP (Tsodyks–Markram) ----------
class STP_TM:
    def __init__(self, U=0.3, tau_rec=600.0, tau_fac=0.0):
        self.U0 = float(U)
        self.tau_rec = float(tau_rec)
        self.tau_fac = float(tau_fac)
        self.R = 1.0
        self.u = float(U)

    def step(self, dt_ms: float, spike: bool) -> float:
        # recover
        self.R += dt_ms * (1.0 - self.R) / self.tau_rec
        self.R = max(0.0, min(1.0, self.R))
        if self.tau_fac > 0:
            self.u += dt_ms * (self.U0 - self.u) / self.tau_fac
            self.u = max(0.0, min(1.0, self.u))
        # consume on spike
        if spike:
            eff = self.u * self.R
            self.R -= eff
            if self.tau_fac > 0:
                self.u += self.U0 * (1.0 - self.u)  # facilitation
                self.u = max(0.0, min(1.0, self.u))
            return eff
        return 0.0

# ---------- single-exp synapse population ----------
class ExpSynPop:
    """
    N identical synapses with exponential decay kernel.
    Each spike adds pre_scale * w_mScm2 (optionally via STP effective release).
    Spatial summation via attenuation vector 'attn' in [0,1].
    """
    def __init__(self, N, tau_ms, w_nS, attn=None, stp=None, seed=0, area_scale=1e-5):
        self.N = int(N)
        self.tau = float(tau_ms)
        self.w_mScm2 = (float(w_nS) * 1e-6) / float(area_scale)  # nS->mS/cm^2
        self.g = np.zeros(self.N, dtype=float)
        self.attn = np.ones(self.N, dtype=float) if attn is None else np.asarray(attn, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.stp = [STP_TM(**stp) for _ in range(self.N)] if stp else None

    def reset(self):
        self.g.fill(0.0)

    def step(self, dt_ms: float, spikes_bool: np.ndarray, pre_scale: float | np.ndarray = 1.0) -> float:
        self.g *= np.exp(-dt_ms / self.tau)
        if self.stp is not None:
            if np.isscalar(pre_scale):
                scale_vec = np.full(self.N, float(pre_scale))
            else:
                scale_vec = np.asarray(pre_scale, dtype=float)
            for i, sp in enumerate(spikes_bool):
                if sp:
                    eff = self.stp[i].step(dt_ms, True)
                    self.g[i] += scale_vec[i] * self.w_mScm2 * eff
                else:
                    self.stp[i].step(dt_ms, False)
        else:
            inc = (self.w_mScm2 * float(pre_scale))
            self.g[spikes_bool] += inc
        return float(np.dot(self.attn, self.g))

# ---------- dual-exp synapse population ----------
class DualExpSynPop:
    """
    Dual exponential (rise, decay) conductance (for NMDA, GABA_B, or any slow conductance).
    """
    def __init__(self, N, tau_rise, tau_decay, w_nS, attn=None, stp=None, seed=0, area_scale=1e-5):
        self.N = int(N)
        self.tau_r = float(tau_rise)
        self.tau_d = float(tau_decay)
        self.w_mScm2 = (float(w_nS) * 1e-6) / float(area_scale)
        self.ar = np.zeros(self.N, dtype=float)
        self.ad = np.zeros(self.N, dtype=float)
        self.attn = np.ones(self.N, dtype=float) if attn is None else np.asarray(attn, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.stp = [STP_TM(**stp) for _ in range(self.N)] if stp else None

    def reset(self):
        self.ar.fill(0.0); self.ad.fill(0.0)

    def step(self, dt_ms: float, spikes_bool: np.ndarray, pre_scale: float | np.ndarray = 1.0) -> float:
        self.ar *= np.exp(-dt_ms / self.tau_r)
        self.ad *= np.exp(-dt_ms / self.tau_d)
        if self.stp is not None:
            if np.isscalar(pre_scale):
                scale_vec = np.full(self.N, float(pre_scale))
            else:
                scale_vec = np.asarray(pre_scale, dtype=float)
            for i, sp in enumerate(spikes_bool):
                if sp:
                    eff = self.stp[i].step(dt_ms, True)
                    inc = scale_vec[i] * self.w_mScm2 * eff
                    self.ar[i] += inc; self.ad[i] += inc
                else:
                    self.stp[i].step(dt_ms, False)
        else:
            inc = (self.w_mScm2 * float(pre_scale))
            self.ar[spikes_bool] += inc
            self.ad[spikes_bool] += inc
        g = (self.ad - self.ar)
        return float(np.dot(self.attn, g))
