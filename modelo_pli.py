from pathlib import Path
import math
import glob
import json
import tempfile
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum

INST_DIR = Path("instances/general")
KS = [3, 4, 5, 6]
TIME_LIMIT_SEC = 10 * 60
OUT_DIR = Path("results")

GUROBI_MEMLIMIT_MB = 16000
GUROBI_THREADS = None
GUROBI_SEED = 42
GUROBI_OUTPUT = 1

SUBPROC_GRACE_SEC = 30


def carregar_pontos(path, limit_n=None):
    cand = [
        (";", ","),
        (",", "."),
        (r"\s+", "."),
        (";", "."),
        (",", ","),
    ]
    for s, d in cand:
        try:
            df = pd.read_csv(
                path,
                sep=s,
                header=None,
                decimal=d,
                engine="python",
                comment="#",
                on_bad_lines="skip",
            )
            if limit_n is not None:
                df = df.iloc[:limit_n, :]
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna().reset_index(drop=True)
            if df.shape[1] >= 2 and len(df) >= 2:
                return df.to_numpy(dtype=float), ""
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    return None, f"Falha ao ler {path} com separadores/decimais conhecidos."


def padronizar_como_R(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=1, keepdims=True)
    sigma[sigma == 0.0] = 1.0
    return (X - mu) / sigma


def distancia_euclidiana_par(X):
    a = np.sum(X * X, axis=1, keepdims=True)
    D2 = a + a.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2, dtype=float)
    np.fill_diagonal(D, 0.0)
    return D


def resolver_kmedoids_ilp(
    D,
    k,
    time_limit,
    mip_gap,
    threads,
    seed,
    memlimit_mb=None,
    output_flag=1,
):
    n = D.shape[0]
    m = Model("kmedoids_ilp")

    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)
    if threads is not None:
        m.Params.Threads = int(threads)
    if seed is not None:
        m.Params.Seed = int(seed)
    if memlimit_mb is not None:
        m.Params.MemLimit = float(memlimit_mb)

    m.Params.OutputFlag = int(output_flag)

    y = m.addVars(n, vtype=GRB.BINARY, name="y")
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

    m.setObjective(
        quicksum(D[i, j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE
    )

    for j in range(n):
        m.addConstr(quicksum(x[i, j] for i in range(n)) == 1, name=f"assign[{j}]")

    for i in range(n):
        for j in range(n):
            m.addConstr(x[i, j] <= y[i], name=f"link[{i},{j}]")

    m.addConstr(quicksum(y[i] for i in range(n)) == k, name="kmedoids")
    for i in range(n):
        m.addConstr(x[i, i] == y[i], name=f"self[{i}]")

    m.optimize()

    status = m.Status
    best_obj = float("nan")
    x_sol = np.zeros((n, n), dtype=int)
    y_sol = np.zeros(n, dtype=int)

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        if m.SolCount > 0:
            best_obj = m.ObjVal
            for i in range(n):
                y_sol[i] = 1 if y[i].X > 0.5 else 0
                for j in range(n):
                    if x[i, j].X > 0.5:
                        x_sol[i, j] = 1

    runtime = m.Runtime

    try:
        mipgap_out = m.MIPGap
    except Exception:
        mipgap_out = math.nan

    try:
        lower_bound = m.ObjBound
    except Exception:
        lower_bound = float("nan")

    upper_bound = float("nan")
    if m.SolCount and math.isfinite(best_obj):
        upper_bound = best_obj

    return status, best_obj, y_sol, x_sol, runtime, mipgap_out, lower_bound, upper_bound


def executar_worker(tmp_in, tmp_out):
    with open(tmp_in, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    D = np.array(cfg["D"], dtype=float)
    k = int(cfg["k"])

    try:
        status, obj, y_sol, x_sol, runtime, mipgap, lb, ub = resolver_kmedoids_ilp(
            D,
            k,
            time_limit=cfg["time_limit"],
            mip_gap=None,
            threads=cfg["threads"],
            seed=cfg["seed"],
            memlimit_mb=cfg["memlimit_mb"],
            output_flag=cfg["output_flag"],
        )
        n = int(D.shape[0])
        avg = obj / n if n > 0 and math.isfinite(obj) else float("nan")
        medoids_1b = (np.where(y_sol == 1)[0] + 1).tolist()

        out = {
            "ok": True,
            "status": status,
            "objective_total": obj,
            "objective_avg_per_point": avg,
            "runtime_sec": runtime,
            "mip_gap": mipgap,
            "lower_bound": lb,
            "upper_bound": ub,
            "medoids_1based": medoids_1b,
            "fail_reason": "",
        }

    except Exception as e:
        out = {
            "ok": False,
            "status": None,
            "objective_total": float("nan"),
            "objective_avg_per_point": float("nan"),
            "runtime_sec": float("nan"),
            "mip_gap": float("nan"),
            "lower_bound": float("nan"),
            "upper_bound": float("nan"),
            "medoids_1based": [],
            "fail_reason": f"EXCEPTION: {type(e).__name__}: {e}",
        }

    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(out, f)


def executar_um_isolado(D, file_name, k):
    cfg = {
        "D": D.tolist(),
        "k": int(k),
        "time_limit": float(TIME_LIMIT_SEC),
        "threads": GUROBI_THREADS,
        "seed": GUROBI_SEED,
        "memlimit_mb": GUROBI_MEMLIMIT_MB,
        "output_flag": GUROBI_OUTPUT,
    }

    with tempfile.TemporaryDirectory() as td:
        tmp_in = Path(td) / "in.json"
        tmp_out = Path(td) / "out.json"

        with open(tmp_in, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        cmd = [sys.executable, __file__, "__worker__", str(tmp_in), str(tmp_out)]
        p = subprocess.Popen(cmd)
        t0 = time.time()

        while True:
            ret = p.poll()
            if ret is not None:
                break
            if (time.time() - t0) > (TIME_LIMIT_SEC + SUBPROC_GRACE_SEC):
                try:
                    p.terminate()
                except Exception:
                    pass
                time.sleep(2)
                try:
                    p.kill()
                except Exception:
                    pass
                ret = -9
                break
            time.sleep(0.2)

        if ret is None:
            ret = p.wait()

        if ret == 0 and tmp_out.exists():
            with open(tmp_out, "r", encoding="utf-8") as f:
                out = json.load(f)
            if out.get("ok", False):
                return {
                    "status_text": "OK",
                    "status": out["status"],
                    "objective_total": out["objective_total"],
                    "objective_avg_per_point": out["objective_avg_per_point"],
                    "runtime_sec": out["runtime_sec"],
                    "mip_gap": out["mip_gap"],
                    "lower_bound": out["lower_bound"],
                    "upper_bound": out["upper_bound"],
                    "medoids_1based": " ".join(map(str, out["medoids_1based"])),
                    "fail_reason": "",
                }
            else:
                return {
                    "status_text": "SOLVER_ERROR",
                    "status": None,
                    "objective_total": float("nan"),
                    "objective_avg_per_point": float("nan"),
                    "runtime_sec": float("nan"),
                    "mip_gap": float("nan"),
                    "lower_bound": float("nan"),
                    "upper_bound": float("nan"),
                    "medoids_1based": "",
                    "fail_reason": out.get("fail_reason", "unknown error"),
                }
        else:
            return {
                "status_text": "FAILED_PROCESS",
                "status": None,
                "objective_total": float("nan"),
                "objective_avg_per_point": float("nan"),
                "runtime_sec": float("nan"),
                "mip_gap": float("nan"),
                "lower_bound": float("nan"),
                "upper_bound": float("nan"),
                "medoids_1based": "",
                "fail_reason": f"exitcode={ret}",
            }


def salvar_acumulado(df_all):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "kmedoids_ilp_all_runs.csv"
    df_all.to_csv(csv_path, index=False)
    try:
        xlsx_path = OUT_DIR / "kmedoids_ilp_all_runs.xlsx"
        df_all.to_excel(xlsx_path, index=False)
    except Exception:
        pass
    return csv_path


def principal():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(INST_DIR / "*.i")) + glob.glob(str(INST_DIR / "*.I")))
    if not files:
        raise SystemExit(f"Nenhuma instância encontrada em {INST_DIR} (.i ou .I)")

    all_rows = []

    for f in files:
        name = Path(f).name
        stem = name[:-2] if name.lower().endswith(".i") else Path(f).stem

        X, err = carregar_pontos(Path(f), limit_n=None)
        if X is None:
            print(err)
            for k in KS:
                row = {
                    "file": name,
                    "stem": stem,
                    "k": k,
                    "n": float("nan"),
                    "status_text": "READ_FAIL",
                    "status": None,
                    "objective_total": float("nan"),
                    "objective_avg_per_point": float("nan"),
                    "runtime_sec": float("nan"),
                    "mip_gap": float("nan"),
                    "lower_bound": float("nan"),
                    "upper_bound": float("nan"),
                    "medoids_1based": "",
                    "fail_reason": err,
                }
                all_rows.append(row)
            df_now = pd.DataFrame(all_rows)
            pth = salvar_acumulado(df_now)
            print(f"[SAVE] Parcial após instância {name}: {pth}")
            continue

        X = padronizar_como_R(X)
        D = distancia_euclidiana_par(X)

        for k in KS:
            res = executar_um_isolado(D, name, k)
            n = D.shape[0]
            row = {
                "file": name,
                "stem": stem,
                "k": k,
                "n": n,
                "status_text": res["status_text"],
                "status": res["status"],
                "objective_total": res["objective_total"],
                "objective_avg_per_point": res["objective_avg_per_point"],
                "runtime_sec": res["runtime_sec"],
                "mip_gap": res["mip_gap"],
                "lower_bound": res["lower_bound"],
                "upper_bound": res["upper_bound"],
                "medoids_1based": res["medoids_1based"],
                "fail_reason": res["fail_reason"],
            }
            all_rows.append(row)

            if res["status_text"] == "OK":
                print(
                    f"{stem} | k={k} | status=OK | total={res['objective_total']:.6f} | "
                    f"avg={res['objective_avg_per_point']:.6f} | time={res['runtime_sec']:.2f}s"
                )
            else:
                print(
                    f"{stem} | k={k} | status={res['status_text']} | reason={res['fail_reason']}"
                )

        df_now = pd.DataFrame(all_rows)
        pth = salvar_acumulado(df_now)
        print(f"[SAVE] Parcial após instância {name}: {pth}")

    df_all = pd.DataFrame(all_rows)
    csv_path = salvar_acumulado(df_all)
    print(f"[OK] Planilha final: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "__worker__":
        executar_worker(sys.argv[2], sys.argv[3])
    else:
        principal()
