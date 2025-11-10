from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_GLOB = "src/problems/kmedoids/solvers/results/grasp_kmedoids_*.csv"
PAPER_CSV = "src/problems/kmedoids/solvers/results/paper_results.csv"
OUTDIR = Path("pp_out_kmedoids")
QUALITY_LEVELS = [1.00, 1.01, 1.05, 1.10, 1.20]
EPS = 1e-12


def _norm_file(s):
    s = str(s).strip()
    return s if s.endswith(".i") else s


def carregar_resultados(csv_glob):
    files = sorted(glob.glob(csv_glob))

    if not files:
        raise SystemExit(f"Nenhum CSV encontrado para o padrão: {csv_glob}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    df["config"] = df["config"].astype(str)
    df["file"] = df["file"].astype(str).map(_norm_file)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")
    df["max_value"] = pd.to_numeric(df["max_value"], errors="coerce")

    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    else:
        df["time_s"] = np.nan

    df["instancia"] = df["file"].astype(str) + "|k=" + df["k"].astype(str)

    return df


def carregar_paper(caminho):
    p = Path(caminho)

    if not p.exists():
        return pd.DataFrame(
            columns=["config", "file", "k", "max_value", "time_s", "instancia"]
        )
    df = pd.read_csv(p)

    df["file"] = df["file"].astype(str).map(_norm_file)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")
    df["s_best"] = pd.to_numeric(df["s_best"], errors="coerce")
    df = df.dropna(subset=["file", "k", "s_best"]).reset_index(drop=True)
    df["config"] = "PAPER"
    df["max_value"] = -df["s_best"]
    df["time_s"] = np.nan
    df["instancia"] = df["file"].astype(str) + "|k=" + df["k"].astype(str)
    return df[["config", "file", "k", "max_value", "time_s", "instancia"]]


def melhor_por_config(df):
    return df.groupby(["instancia", "config"], as_index=False).agg(
        best_value=("max_value", "max"), best_time=("time_s", "min")
    )


def calcular_base_B(df_cfg):
    return (
        df_cfg.groupby("instancia", as_index=False)["best_value"]
        .max()
        .rename(columns={"best_value": "B_i"})
    )


def calcular_ratios(df_cfg, B):
    d = df_cfg.merge(B, on="instancia", how="right")

    linhas = []
    for inst, g in d.groupby("instancia"):
        Bi = g["B_i"].iloc[0]
        denom = max(EPS, abs(Bi)) if pd.notna(Bi) else EPS
        for _, row in g.iterrows():
            vi = row["best_value"]
            r = np.inf if (pd.isna(Bi) or pd.isna(vi)) else 1.0 + (Bi - vi) / denom
            linhas.append({"instancia": inst, "metodo": row["config"], "r": float(r)})

    return pd.DataFrame(linhas)


def gerar_curvas(df_ratios, taus):
    insts = sorted(df_ratios["instancia"].unique())
    n = float(len(insts)) if insts else 1.0
    blocos = []

    for m, g in df_ratios.groupby("metodo"):
        r = g["r"].to_numpy()
        rho = [(r <= t).sum() / n for t in taus]
        blocos.append(pd.DataFrame({"tau": taus, "rho": rho, "metodo": m}))

    return pd.concat(blocos, ignore_index=True)


def salvar_csv(caminho, df):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(df.to_csv(index=False), encoding="utf-8")


def plotar_curvas(df_curvas, titulo, png, xlim=None, caption=None):
    labels = sorted(df_curvas["metodo"].unique())
    n = len(labels)
    ncol = 3 if n >= 3 else n
    rows = int(np.ceil(n / max(1, ncol)))
    yoff = -0.18 - 0.08 * (rows - 1)

    if caption:
        titulo = f"{titulo}\n{caption}"

    plt.figure(figsize=(6.6, 4.8))
    plt.ylim(0.0, 1.03)

    for m in labels:
        g = df_curvas[df_curvas["metodo"] == m].sort_values("tau")
        plt.step(g["tau"].values, g["rho"].values, where="post", label=m)

    if xlim is not None:
        plt.xlim(*xlim)
        if xlim[0] >= 1.0 and xlim[1] <= 2.0:
            ticks = [
                t for t in [1.00, 1.01, 1.05, 1.10, 1.20] if xlim[0] <= t <= xlim[1]
            ]
            if ticks:
                plt.xticks(ticks, [f"{t:.2f}" for t in ticks])

    plt.grid(True, alpha=0.35)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\rho(\tau)$")
    plt.title(titulo)
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, yoff),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        handlelength=2.5,
        columnspacing=1.2,
    )

    plt.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()


def resumir_niveis(df_ratios, niveis):
    insts = sorted(df_ratios["instancia"].unique())
    n = float(len(insts)) if insts else 1.0
    linhas = []
    for m, g in df_ratios.groupby("metodo"):
        r = g["r"].to_numpy()
        linha = {"metodo": m}
        for lv in niveis:
            linha[f"rho@{lv:.2f}"] = float((r <= lv).sum()) / n
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("metodo").reset_index(drop=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df_raw = carregar_resultados(CSV_GLOB)
    df_paper = carregar_paper(PAPER_CSV)
    if not df_paper.empty:
        df_raw = pd.concat([df_raw, df_paper], ignore_index=True)
    df_cfg = melhor_por_config(df_raw)
    B = calcular_base_B(df_cfg)
    ratios = calcular_ratios(df_cfg, B)
    salvar_csv(OUTDIR / "quality_ratios_all.csv", ratios)
    fin = ratios[np.isfinite(ratios["r"])]
    tau_max = max(1.02, float(fin["r"].max())) if not fin.empty else 1.02
    taus_full = np.linspace(1.0, tau_max, 200)
    curvas_full = gerar_curvas(ratios, taus_full)
    salvar_csv(OUTDIR / "quality_curves_full_all.csv", curvas_full)
    plotar_curvas(
        curvas_full,
        "Performance Profile (todas as instâncias)",
        OUTDIR / "quality_full_all.png",
        xlim=(1.0, tau_max),
        caption=None,
    )
    for perc in [1.01, 1.05, 1.10, 1.20]:
        taus_zoom = np.linspace(1.0, perc, 200)
        curvas_zoom = gerar_curvas(ratios, taus_zoom)
        salvar_csv(
            OUTDIR / f"quality_curves_zoom{int((perc-1)*100):02d}_all.csv", curvas_zoom
        )
        plotar_curvas(
            curvas_zoom,
            f"Performance Profile — até {int((perc-1)*100)}%",
            OUTDIR / f"quality_zoom{int((perc-1)*100):02d}_all.png",
            xlim=(1.0, perc),
            caption=None,
        )
    resumo = resumir_niveis(ratios, QUALITY_LEVELS)
    salvar_csv(OUTDIR / "quality_summary_all.csv", resumo)
    mapa_inst_k = df_raw[["instancia", "k", "file"]].drop_duplicates()
    ratios_k = ratios.merge(mapa_inst_k, on="instancia", how="left")
    for k in sorted(ratios_k["k"].dropna().unique()):
        rk = ratios_k[ratios_k["k"] == k].copy()
        if rk.empty:
            continue
        cap = "instâncias: " + ", ".join(sorted(rk["instancia"].unique()))
        fin_k = rk[np.isfinite(rk["r"])]
        tau_max_k = max(1.02, float(fin_k["r"].max())) if not fin_k.empty else 1.02
        curvas_k = gerar_curvas(
            rk[["instancia", "metodo", "r"]], np.linspace(1.0, tau_max_k, 200)
        )
        salvar_csv(OUTDIR / f"k_{int(k)}/quality_curves_full_k{int(k)}.csv", curvas_k)
        plotar_curvas(
            curvas_k,
            f"Performance Profile (k={int(k)})",
            OUTDIR / f"k_{int(k)}/quality_full_k{int(k)}.png",
            xlim=(1.0, tau_max_k),
            caption=cap,
        )
        for perc in [1.01, 1.05, 1.10, 1.20]:
            curvas_zoom_k = gerar_curvas(
                rk[["instancia", "metodo", "r"]], np.linspace(1.0, perc, 200)
            )
            salvar_csv(
                OUTDIR
                / f"k_{int(k)}/quality_curves_zoom{int((perc-1)*100):02d}_k{int(k)}.csv",
                curvas_zoom_k,
            )
            plotar_curvas(
                curvas_zoom_k,
                f"Performance Profile — até {int((perc-1)*100)}% (k={int(k)})",
                OUTDIR
                / f"k_{int(k)}/quality_zoom{int((perc-1)*100):02d}_k{int(k)}.png",
                xlim=(1.0, perc),
                caption=cap,
            )
    for inst in sorted(ratios["instancia"].unique()):
        ri = ratios[ratios["instancia"] == inst]
        fin_i = ri[np.isfinite(ri["r"])]
        tau_max_i = max(1.02, float(fin_i["r"].max())) if not fin_i.empty else 1.02
        curvas_i = gerar_curvas(ri, np.linspace(1.0, tau_max_i, 200))
        f, k = inst.split("|k=")
        salvar_csv(OUTDIR / f"inst_{f}_k{k}/quality_curves_full_{f}_k{k}.csv", curvas_i)
        plotar_curvas(
            curvas_i,
            f"Performance Profile ({f} | k={k})",
            OUTDIR / f"inst_{f}_k{k}/quality_full_{f}_k{k}.png",
            xlim=(1.0, tau_max_i),
            caption=inst,
        )
    print("Pronto! Saídas em:", OUTDIR)


if __name__ == "__main__":
    main()
