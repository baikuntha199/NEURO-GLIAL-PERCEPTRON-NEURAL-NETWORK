import yaml
from src.network import StackedNet

def test_build_and_run_smoke():
    # Tiny config for CI-speed smoke
    cfg = {
        "sim": {"T_ms": 200, "dt": 0.05, "seed": 1},
        "mode": {"neuron": "lif", "glia_scope": "layer"},
        "neuron": {"C":1.0,"gNa":120,"gK":36,"gL":0.3,"ENa":50,"EK":-77,"EL":-54.4,
                   "gCa":0.5,"ECa":120,"gKCa":2.0,"Ca_rest":0.05,"tau_Ca":80,"alpha_Ca":0.002,"Kd_KCa":0.3},
        "geometry": {"soma_area_cm2":1e-5,"dend_area_cm2":2e-5,"g_axial_mScm2":0.5},
        "synapses": {
            "ampa":{"w_nS":0.8,"tau_ms":5},
            "nmda":{"w_nS":0.15,"tau_rise":5,"tau_decay":80,"mg_mM":1.0},
            "gabaA":{"w_nS":0.7,"tau_ms":10,"E_mV":-70},
            "gabaB":{"w_nS":0.3,"tau_rise":50,"tau_decay":200,"E_mV":-95}
        },
        "stp": {"enabled": True, "U":0.3,"tau_rec":600,"tau_fac":0},
        "glia":{"enabled":True,"tau_ip3":500,"tau_ca":800,"k_syn":0.015,"k_ip3":0.02,
                "theta":0.15,"slope":0.08,"ca_noise":0.0,
                "pre_exc_gain":-0.2,"pre_inh_gain":0.1,"post_exc_gain":-0.2,"post_inh_gain":0.2},
        "layer": {"n_layers": 2, "n_neurons": 20, "p_connect": 0.2},
        "input": {"n_channels": 40, "base_rate_hz": 3.0, "stim":{"on_ms":50,"off_ms":150,"rate_exc_hz":12.0}},
        "decoder": {"window_ms": 20, "train_frac": 0.6}
    }
    net = StackedNet(cfg)
    arr = net.run()
    assert arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] == cfg["layer"]["n_neurons"]
