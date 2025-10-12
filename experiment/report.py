# make_plots.py
import os, csv, json, math
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALG_ORDER = ["ISTA", "FISTA", "IHT", "OMP", "SP", "CoSaMP"]

CASE_ORDER = [
    ("gaussian", 0.000),
    ("gaussian", 0.010),
    ("bernoulli", 0.000),
    ("bernoulli", 0.010),
]
CASE_LABEL = {
    ("gaussian", 0.000): "Gaussian, noise=0.000",
    ("gaussian", 0.010): "Gaussian, noise=0.010",
    ("bernoulli", 0.000): "Bernoulli, noise=0.000",
    ("bernoulli", 0.010): "Bernoulli, noise=0.010",
}
CASE_COLOR = {
    ("gaussian", 0.000): "C0",
    ("gaussian", 0.010): "C1",
    ("bernoulli", 0.000): "C2",
    ("bernoulli", 0.010): "C3",
}


def _parse_noise(dirname: str):
    if not dirname.startswith("noise_"):
        return None
    try:
        return float(dirname.split("_", 1)[1])
    except Exception:
        return None


def collect_cases_from_json():
    data = {}
    cases = set()
    for algo in ALG_ORDER:
        algo_dir = os.path.join(RESULTS_DIR, algo)
        if not os.path.isdir(algo_dir):
            continue
        for measure in os.listdir(algo_dir):
            measure_dir = os.path.join(algo_dir, measure)
            if not os.path.isdir(measure_dir):
                continue
            for noise_tag in os.listdir(measure_dir):
                noise_rel = _parse_noise(noise_tag)
                if noise_rel is None:
                    continue
                case_dir = os.path.join(measure_dir, noise_tag)
                sum_path = os.path.join(case_dir, "summary.json")
                if not os.path.isfile(sum_path):
                    continue
                try:
                    with open(sum_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    psnr = float(meta.get("metrics", {}).get("PSNR", float("nan")))
                    elapsed = float(meta.get("elapsed_sec", float("nan")))
                    iters = meta.get("iters", None)
                    if iters is None:
                        iters = float("nan")
                    else:
                        try:
                            iters = int(iters)
                        except Exception:
                            iters = float("nan")
                except Exception:
                    continue

                key = (measure, noise_rel)
                cases.add(key)
                data.setdefault(key, {})[algo] = {
                    "PSNR": psnr,
                    "elapsed_sec": elapsed,
                    "iters": iters,
                }
    return cases, data


def plot_bar(metric_name, data_pairs, title, save_path, figsize=(9, 5), dpi=200):
    order_map = {a: i for i, a in enumerate(ALG_ORDER)}
    data_pairs.sort(key=lambda p: order_map.get(p[0], 999))
    algs = [a for a, _ in data_pairs]
    vals = [v for _, v in data_pairs]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(algs))
    ax.bar(x, vals)
    ax.set_xticks(x, algs)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.margins(x=0.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_grouped_bars_all_cases(
    metric_name, data, title, save_path, figsize=(11, 6), dpi=220
):
    algs = ALG_ORDER[:]
    x = np.arange(len(algs))
    num_cases = len(CASE_ORDER)

    width = min(0.8 / max(num_cases, 1), 0.18)
    gap = 0.02
    offsets = (np.arange(num_cases) - (num_cases - 1) / 2.0) * (width + gap)

    fig, ax = plt.subplots(figsize=figsize)
    plotted_any = False

    for i, case in enumerate(CASE_ORDER):
        measure, noise_rel = case
        label = CASE_LABEL.get(case, f"{measure}, noise={noise_rel:.3f}")
        color = CASE_COLOR.get(case, None)

        vals = []
        per_case = data.get((measure, noise_rel), {})
        for a in algs:
            v = np.nan
            if a in per_case:
                val = per_case[a].get(metric_name, np.nan)
                v = val if (val is not None) else np.nan
            vals.append(v)
        vals = np.array(vals, dtype=float)
        if np.all(np.isnan(vals)):
            continue

        ax.bar(x + offsets[i], vals, width=width, label=label, color=color)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xticks(x, algs)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.margins(x=0.05)

    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return True


def main():
    cases, data = collect_cases_from_json()
    if not cases:
        raise FileNotFoundError(
            "Không tìm thấy bất kỳ summary.json nào trong ./results/<ALGO>/<measure>/noise_*/"
        )

    for measure, noise_rel in sorted(cases, key=lambda x: (x[0], x[1])):
        per_algo = data.get((measure, noise_rel), {})

        # PSNR
        psnr_pairs = [
            (a, v["PSNR"]) for a, v in per_algo.items() if not math.isnan(v["PSNR"])
        ]
        if psnr_pairs:
            save_path = os.path.join(
                FIG_DIR, f"psnr_{measure}_noise_{noise_rel:.3f}.png"
            )
            plot_bar(
                "PSNR (dB)",
                psnr_pairs,
                f"PSNR — {measure}, noise={noise_rel:.3f}",
                save_path,
            )

        # Runtime
        time_pairs = [
            (a, v["elapsed_sec"])
            for a, v in per_algo.items()
            if not math.isnan(v["elapsed_sec"])
        ]
        if time_pairs:
            save_path = os.path.join(
                FIG_DIR, f"time_runtime_{measure}_noise_{noise_rel:.3f}.png"
            )
            plot_bar(
                "Thời gian chạy (s)",
                time_pairs,
                f"Runtime — {measure}, noise={noise_rel:.3f}",
                save_path,
            )

        # Iterations
        iters_pairs = []
        for a, v in per_algo.items():
            itv = v["iters"]
            if isinstance(itv, (int, np.integer)) and not math.isnan(itv):
                iters_pairs.append((a, int(itv)))
        if iters_pairs:
            save_path = os.path.join(
                FIG_DIR, f"time_iters_{measure}_noise_{noise_rel:.3f}.png"
            )
            plot_bar(
                "Số vòng hội tụ (iters)",
                iters_pairs,
                f"Iterations — {measure}, noise={noise_rel:.3f}",
                save_path,
            )

    save_path = os.path.join(FIG_DIR, f"psnr_all_cases.png")
    plot_grouped_bars_all_cases("PSNR", data, "PSNR — All cases", save_path)

    save_path = os.path.join(FIG_DIR, f"time_runtime_all_cases.png")
    plot_grouped_bars_all_cases("elapsed_sec", data, "Runtime — All cases", save_path)

    save_path = os.path.join(FIG_DIR, f"time_iters_all_cases.png")
    plot_grouped_bars_all_cases("iters", data, "Iterations — All cases", save_path)

    print(f"Đã lưu hình vào: {FIG_DIR}")
    print("  - psnr_<measure>_noise_<...>.png  &  psnr_all_cases.png")
    print("  - time_runtime_<measure>_noise_<...>.png  &  time_runtime_all_cases.png")
    print("  - time_iters_<measure>_noise_<...>.png  &  time_iters_all_cases.png")


if __name__ == "__main__":
    main()
