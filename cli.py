import argparse, yaml
import numpy as np
import matplotlib.pyplot as plt

from .network import StackedNet
from .decoder import window_counts_from_voltages, indexes_for_on_off, decode_stim_on_off

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    net = StackedNet(cfg)
    last_layer_time_series = net.run()   # (steps, N_last)

    dt = float(cfg["sim"]["dt"])
    T  = float(cfg["sim"]["T_ms"])
    window = float(cfg["decoder"]["window_ms"])
    X = window_counts_from_voltages(last_layer_time_series, dt, window)

    # labels from input (if present)
    if "input" in cfg:
        on  = float(cfg["input"]["stim"]["on_ms"])
        off = float(cfg["input"]["stim"]["off_ms"])
        on_idx = indexes_for_on_off(T, dt, window, on, off)
        acc, auc = decode_stim_on_off(X, on_idx, cfg["decoder"]["train_frac"])
        print(f"[Decoder] ACC={acc:.3f}  AUC={auc:.3f}")
    else:
        print("[Decoder] No input.stim in config; skipping classification.")

    # quick plot
    t = np.arange(last_layer_time_series.shape[0]) * dt
    show = min(8, last_layer_time_series.shape[1])
    plt.figure(figsize=10,4)
    plt.plot(t, last_layer_time_series[:, :show])
    if "input" in cfg:
        plt.axvspan(on, off, color="lightgray", alpha=0.5, label="Stimulus")
        plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage / Rate (a.u.)")
    plt.title("Last-layer activity (subset)")
    plt.tight_layout()
    plt.show()
