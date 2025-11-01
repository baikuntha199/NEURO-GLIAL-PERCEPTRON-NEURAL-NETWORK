import numpy as np

def _sigmoid(x):  # stable logistic
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

class Astrocyte:
    """
    Minimal astrocyte with IP3–Ca2+ reservoir.
    Senses local excitatory activity (e.g., g_AMPA + g_NMDA) and
    returns pre/post gains for exc/inh pathways.
    """
    def __init__(self, cfg: dict, rng=None):
        g = cfg.get('glia', {})
        self.enabled = bool(g.get('enabled', True))
        self.tau_ip3 = float(g.get('tau_ip3', 500.0))
        self.tau_ca  = float(g.get('tau_ca', 800.0))
        self.k_syn   = float(g.get('k_syn', 0.015))
        self.k_ip3   = float(g.get('k_ip3', 0.02))
        self.theta   = float(g.get('theta', 0.15))
        self.slope   = float(g.get('slope', 0.08))
        self.ca_noise= float(g.get('ca_noise', 0.0005))
        self.pre_exc_gain  = float(g.get('pre_exc_gain', -0.35))
        self.pre_inh_gain  = float(g.get('pre_inh_gain',  0.10))
        self.post_exc_gain = float(g.get('post_exc_gain', -0.25))
        self.post_inh_gain = float(g.get('post_inh_gain',  0.20))
        self.IP3 = 0.0
        self.Ca  = 0.0
        self.rng = np.random.default_rng() if rng is None else rng

    def step(self, dt_ms: float, activity: float):
        if not self.enabled:
            return dict(pre_exc=1.0, pre_inh=1.0, post_exc=1.0, post_inh=1.0, Ca=self.Ca)
        # IP3
        self.IP3 += dt_ms * ( self.k_syn * max(0.0, float(activity)) - self.IP3 / self.tau_ip3 )
        self.IP3 = max(0.0, self.IP3)
        # Ca
        noise = self.ca_noise * self.rng.normal()
        self.Ca += dt_ms * ( self.k_ip3 * self.IP3 * (1.0 - self.Ca) - self.Ca / self.tau_ca ) + noise
        self.Ca = min(max(self.Ca, 0.0), 1.0)
        # gate
        G = _sigmoid((self.Ca - self.theta) / max(1e-6, self.slope))
        pre_exc  = max(0.0, 1.0 + self.pre_exc_gain  * G)
        pre_inh  = max(0.0, 1.0 + self.pre_inh_gain  * G)
        post_exc = max(0.0, 1.0 + self.post_exc_gain * G)
        post_inh = max(0.0, 1.0 + self.post_inh_gain * G)
        return dict(pre_exc=pre_exc, pre_inh=pre_inh, post_exc=post_exc, post_inh=post_inh, Ca=self.Ca)
