import numpy as np
from typing import Literal, Tuple

from .neuron_bio import BioNeuron
from .neuron_light import LIFNeuron, AdExNeuron
from .neuron_rate import RateUnit
from .synapses import ExpSynPop, DualExpSynPop
from .glia import Astrocyte

ModeNeuron = Literal["bio", "lif", "adex", "rate"]
GliaScope  = Literal["neuron", "layer"]

class Layer:
    """
    A layer of neurons with dendritic synapses (spiking modes) or rate units.
    Supports astrocyte modulation at neuron- or layer-scope.
    """
    def __init__(self, cfg: dict, mode_neuron: ModeNeuron, glia_scope: GliaScope,
                 n_inputs: int, n_neurons: int, seed: int):
        self.cfg = cfg
        self.dt  = float(cfg["sim"]["dt"])
        self.mode = mode_neuron
        self.scope = glia_scope
        self.Nin = int(n_inputs)
        self.N   = int(n_neurons)
        self.rng = np.random.default_rng(seed)

        # --- Astrocyte(s) ---
        if self.scope == "layer":
            self.astro = Astrocyte(cfg, rng=self.rng)
        else:
            self.astro = [Astrocyte(cfg, rng=np.random.default_rng(seed+i)) for i in range(self.N)]

        # --- Build neurons ---
        if self.mode == "bio":
            geom = cfg["geometry"]
            chan = cfg["neuron"]
            self.neurons = [BioNeuron(dt=self.dt, geom=geom, **chan) for _ in range(self.N)]
        elif self.mode == "lif":
            self.neurons = [LIFNeuron(dt=self.dt) for _ in range(self.N)]
        elif self.mode == "adex":
            self.neurons = [AdExNeuron(dt=self.dt) for _ in range(self.N)]
        elif self.mode == "rate":
            self.neurons = [RateUnit(tau=10.0) for _ in range(self.N)]
        else:
            raise ValueError("Unknown mode")

        # --- Synapse populations (spiking modes only) ---
        if self.mode in ("bio", "lif", "adex"):
            syn = cfg["synapses"]
            stp = cfg["stp"] if cfg["stp"]["enabled"] else None
            area_soma = cfg["geometry"]["soma_area_cm2"]
            area_dend = cfg["geometry"]["dend_area_cm2"]
            attn = np.linspace(1.0, 0.4, self.Nin)

            self.ampa = [ExpSynPop(self.Nin, syn["ampa"]["tau_ms"], syn["ampa"]["w_nS"],
                                   attn=attn, stp=stp, seed=seed+10+i, area_scale=area_dend)
                         for i in range(self.N)]
            self.nmda = [DualExpSynPop(self.Nin, syn["nmda"]["tau_rise"], syn["nmda"]["tau_decay"],
                                       syn["nmda"]["w_nS"], attn=attn, stp=stp, seed=seed+110+i, area_scale=area_dend)
                         for i in range(self.N)]
            self.gabaA= [ExpSynPop(self.Nin, syn["gabaA"]["tau_ms"], syn["gabaA"]["w_nS"],
                                   attn=attn, stp=None, seed=seed+210+i, area_scale=area_dend)
                         for i in range(self.N)]
            self.gabaB= [DualExpSynPop(self.Nin, syn["gabaB"]["tau_rise"], syn["gabaB"]["tau_decay"],
                                       syn["gabaB"]["w_nS"], attn=attn, stp=None, seed=seed+310+i, area_scale=area_dend)
                         for i in range(self.N)]
            # init pre-scale memories
            self.pre_exc = np.ones(self.N)
            self.pre_inh = np.ones(self.N)

    # ---- external drive API (first layer) ----
    def step_with_external(self, exc_spikes: np.ndarray, inh_spikes: np.ndarray):
        """
        exc_spikes / inh_spikes: shape (N, Nin) boolean for each postsyn neuron.
        Returns: tuple (voltage_or_rate, spike_proxy_or_none)
        """
        if self.mode in ("bio", "lif", "adex"):
            Vs = np.empty(self.N)
            spike_proxy = np.zeros(self.N, dtype=bool)

            # layer-scope astro pre-gains for this tick (if any)
            if self.scope == "layer":
                # approximate activity with mean excitatory g from previous tick if available
                # or estimate from number of spikes; here we compute after raw g’s
                layer_pre_exc = layer_pre_inh = 1.0
                layer_post_exc = layer_post_inh = 1.0

            for i in range(self.N):
                # --- pre-syn increments (use last pre_* scales) ---
                gA_raw = self.ampa[i].step(self.dt, exc_spikes[i], pre_scale=self.pre_exc[i])
                gN_raw = self.nmda[i].step(self.dt, exc_spikes[i], pre_scale=self.pre_exc[i])
                gGA_raw= self.gabaA[i].step(self.dt, inh_spikes[i], pre_scale=self.pre_inh[i])
                gGB_raw= self.gabaB[i].step(self.dt, inh_spikes[i], pre_scale=self.pre_inh[i])

                if self.scope == "neuron":
                    mod = self.astro[i].step(self.dt, activity=max(0.0, gA_raw + gN_raw))
                    gA = gA_raw * mod["post_exc"]; gN = gN_raw * mod["post_exc"]
                    gGA= gGA_raw * mod["post_inh"]; gGB= gGB_raw * mod["post_inh"]
                    self.pre_exc[i] = mod["pre_exc"]; self.pre_inh[i] = mod["pre_inh"]
                else:
                    # temp store raw to compute layer activity
                    gA, gN, gGA, gGB = gA_raw, gN_raw, gGA_raw, gGB_raw
                    # scales applied after we get layer mod below
                    self._cache_last = self._cache_last if hasattr(self, "_cache_last") else {}
                    self._cache_last[i] = (gA, gN, gGA, gGB)

                # deliver to neuron
                if self.mode == "bio":
                    self.neurons[i].set_syn_dend(gA, gN, gGA, gGB)
                    Vs[i], _ = self.neurons[i].step(Iext_soma=0.0)
                    spike_proxy[i] = (Vs[i] >= 0.0)  # simple 0-cross proxy
                elif self.mode == "lif":
                    I_syn = (gA + gN) - (gGA + gGB)  # crude drive
                    V, spk = self.neurons[i].step(I_syn=I_syn)
                    Vs[i] = V; spike_proxy[i] = spk
                elif self.mode == "adex":
                    I_syn = (gA + gN) - (gGA + gGB)
                    V, spk = self.neurons[i].step(I_syn=I_syn)
                    Vs[i] = V; spike_proxy[i] = spk

            if self.scope == "layer":
                # update one astro from mean activity then apply post + pre scales next step
                gA_mean = np.mean([self._cache_last[i][0] for i in range(self.N)]) if self.N>0 else 0.0
                gN_mean = np.mean([self._cache_last[i][1] for i in range(self.N)]) if self.N>0 else 0.0
                modL = self.astro.step(self.dt, activity=max(0.0, gA_mean + gN_mean))
                # apply post gains retrospectively for this step (approx via scaling Vs proxy)
                # For simplicity we’ll update pre-scales for next tick:
                self.pre_exc[:] = modL["pre_exc"]; self.pre_inh[:] = modL["pre_inh"]
                # (post gains were not applied this tick to keep code light)

            return Vs, spike_proxy

        else:  # rate
            # drive = (W @ prev_r) will be handled by network; here we just pass-through external as current
            rates = np.empty(self.N)
            for i, unit in enumerate(self.neurons):
                # approximate external drive strength by (#exc - #inh) spikes this step
                drive = float(exc_spikes[i].sum() - inh_spikes[i].sum())
                # simple layer astro modulation of gain
                if self.scope == "layer":
                    if not hasattr(self, "_layer_astate"):
                        self._layer_astate = 0.0
                    mod = self.astro.step(self.dt, activity=max(0.0, drive))
                    gain = mod["post_exc"]
                else:
                    mod = self.astro[i].step(self.dt, activity=max(0.0, drive))
                    gain = mod["post_exc"]
                rates[i] = unit.step(self.dt, gain * drive)
            return rates, None

    # ---- internal drive API (used by deeper layers) ----
    def step_with_internal(self, exc_mat_bool: np.ndarray, inh_mat_bool: np.ndarray):
        return self.step_with_external(exc_mat_bool, inh_mat_bool)
