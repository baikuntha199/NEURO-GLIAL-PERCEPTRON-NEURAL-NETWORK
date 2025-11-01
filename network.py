import numpy as np
import yaml

from .layer import Layer
from .utils import set_seed

class StackedNet:
    """
    Stacked feedforward network with optional lateral inhibition fraction.
    First layer receives external Poisson drive (stim on/off).
    Higher layers receive spike proxies from previous layer via random mask.
    Works for spiking (bio/lif/adex) and rate modes.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt = float(cfg["sim"]["dt"])
        self.seed = int(cfg["sim"]["seed"])
        self.rng = set_seed(self.seed)

        self.mode_neuron = cfg["mode"]["neuron"]
        self.glia_scope  = cfg["mode"]["glia_scope"]

        L = int(cfg["layer"]["n_layers"])
        N = int(cfg["layer"]["n_neurons"])
        p = float(cfg["layer"]["p_connect"])

        self.n_inputs = int(cfg["input"]["n_channels"]) if "input" in cfg else N
        self.layers = []

        # Build layers
        for ell in range(L):
            nin = self.n_inputs if ell == 0 else N
            ly = Layer(cfg, self.mode_neuron, self.glia_scope, n_inputs=nin, n_neurons=N, seed=self.seed+1000*ell)
            self.layers.append(ly)

        # Feedforward masks (for spiking modes): L_{k-1} -> L_k
        self.ff_masks = []
        for ell in range(1, L):
            mask = (self.rng.random((N, N)) < p)
            self.ff_masks.append(mask)

    def run(self):
        T_ms = float(self.cfg["sim"]["T_ms"])
        steps = int(T_ms / self.dt)

        # Input construction (Poisson to layer-0)
        if "input" in self.cfg:
            inp = self.cfg["input"]
            base = np.ones(self.n_inputs) * float(inp["base_rate_hz"])
            on_ms  = float(inp["stim"]["on_ms"])
            off_ms = float(inp["stim"]["off_ms"])
            rate_hi = float(inp["stim"]["rate_exc_hz"])
        else:
            base = np.zeros(self.n_inputs); on_ms, off_ms = 0, 0; rate_hi = 0

        # Recording small subset (L2 or last layer voltages/rates)
        rec = []

        # Spike proxy carry for connecting layers
        prev_proxy = None

        for k in range(steps):
            t = k * self.dt

            # ---- layer 0 external spikes ----
            rate_exc = rate_hi if (t >= on_ms and t <= off_ms) else base
            lam = rate_exc * (self.dt/1000.0)
            exc0 = (self.rng.random((self.layers[0].N, self.n_inputs)) < lam)  # (N, Nin)
            inh0 = np.zeros_like(exc0, dtype=bool)

            out0, proxy0 = self.layers[0].step_with_external(exc0, inh0)
            prev_proxy = proxy0

            # ---- deeper layers ----
            for ell in range(1, len(self.layers)):
                N = self.layers[ell].N
                NinPrev = self.layers[ell-1].N

                exc = np.zeros((N, NinPrev), dtype=bool)
                inh = np.zeros_like(exc)

                if prev_proxy is not None:  # spiking modes
                    mask = self.ff_masks[ell-1]
                    # project previous spike proxies to boolean input matrix
                    for j in range(N):
                        src = mask[j]
                        if src.any():
                            exc[j, src] = prev_proxy[src]
                else:
                    # rate mode: convert prev "rates" to Bernoulli spikes for compatibility
                    prev_rates = out0 if ell == 1 else rec[-1]  # crude
                    p_spk = np.clip(prev_rates * 0.01, 0.0, 1.0)
                    randm = self.rng.random((N, NinPrev))
                    exc = randm < p_spk

                outk, prox = self.layers[ell].step_with_internal(exc, inh)
                prev_proxy = prox

            rec.append(outk)  # record last layer output each step

        return np.array(rec)  # shape (steps, N_last)
