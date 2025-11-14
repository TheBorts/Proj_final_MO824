from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATHS = [
    "results/ttt_grasp_kmedoids_5_easy.csv",
    "results/ttt_grasp_kmedoids_25_easy.csv",
    "results/ttt_grasp_kmedoids_5_hard.csv",
    "results/ttt_grasp_kmedoids_25_hard.csv",

    # "results/ttt_outro.csv",
]

# 2) Diretório de saída dos PNGs
OUTPUT_DIR = "plots_ttt"

# 3) Filtros opcionais (use None para não filtrar)
ONLY_INST = None                 # ex.: ["haberman.i", "iris.i"]
ONLY_K = None                    # ex.: [3, 5]
ONLY_CONFIG_PREFIX = None        # ex.: ["GRASP_alpha=0.05", "RPG_p=10"]
EXCLUDE_CONFIG_PREFIX = None     # ex.: ["RPG_p=20"]

# 4) Texto opcional no título dos gráficos
TITLE_SUFFIX = ""                # ex.: "GRASP study (1 min/run, 50 runs)"



def compute_ttt_empirical(times_ms: np.ndarray):
    n = len(times_ms)
    if n == 0:
        return np.array([]), np.array([])
    x = np.sort(times_ms)
    y = np.array([(i + 0.5) / n for i in range(n)], dtype=float)
    return x, y


def load_ttt_csvs(paths):
    dfs = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            print(f"[WARN] CSV não encontrado: {pth}")
            continue
        df = pd.read_csv(pth)
        required = {"instance", "file", "k", "config", "target_avg", "run_idx", "time_to_target_ms"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {pth} sem colunas {missing}. Esperado: {sorted(required)}")
        df["__src"] = pth.name
        dfs.append(df)
        print(f"[OK] Carregado: {pth} ({len(df)} linhas)")
    if not dfs:
        raise SystemExit("Nenhum CSV válido foi carregado.")
    return pd.concat(dfs, ignore_index=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if ONLY_INST:
        out = out[out["file"].isin(ONLY_INST)]
    if ONLY_K:
        out = out[out["k"].isin(ONLY_K)]
    if ONLY_CONFIG_PREFIX:
        m = np.zeros(len(out), dtype=bool)
        for pref in ONLY_CONFIG_PREFIX:
            m |= out["config"].str.startswith(pref)
        out = out[m]
    if EXCLUDE_CONFIG_PREFIX:
        m = np.ones(len(out), dtype=bool)
        for pref in EXCLUDE_CONFIG_PREFIX:
            m &= ~out["config"].str.startswith(pref)
        out = out[m]
    return out


def plot_ttt_group(df_pair, instance_file, k, target, output_dir, title_suffix=""):
    plt.figure(figsize=(8, 6))

    configs = sorted(df_pair["config"].unique())
    for cfg in configs:
        sub = df_pair[df_pair["config"] == cfg].copy()

        success = sub[sub["time_to_target_ms"] >= 0]["time_to_target_ms"].to_numpy(dtype=float)
        total_runs = len(sub)
        succ_runs = success.size

        xs, ys = compute_ttt_empirical(success)
        label = f"{cfg} (succ {succ_runs}/{total_runs})"

        if xs.size > 0:
            plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, label=label, alpha=0.9)
        else:
            plt.plot([], [], label=label)

    plt.xlabel("Time to target (ms)")
    plt.ylabel("Cumulative probability")
    title = f"TTT — file={instance_file}, k={k}, target={target}"
    if title_suffix:
        title += f" — {title_suffix}"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(left=0)
    plt.legend(loc="lower right", framealpha=0.9)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_inst = Path(instance_file).name.replace("/", "_")
    out = Path(output_dir) / f"ttt_{safe_inst}_k{k}_{str(target).replace('.', '_')}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot salvo: {out}")


def main():
    df = load_ttt_csvs(CSV_PATHS)
    df = apply_filters(df)

    if df.empty:
        raise SystemExit("Nenhuma linha após filtros — nada para plotar.")

    groups = df.groupby(["file", "k", "target_avg"])
    print(f"[INFO] Pares (file,k,target): {len(groups)}")

    for (file_name, k, target), df_pair in groups:
        plot_ttt_group(df_pair, file_name, k, target, OUTPUT_DIR, TITLE_SUFFIX)


if __name__ == "__main__":
    main()
