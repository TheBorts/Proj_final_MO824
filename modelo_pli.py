import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum


def load_points(path, sep=";", decimal=",", limit_n=None):
    df = pd.read_csv(
        path, sep=sep, header=None, decimal=decimal, engine="python", comment="#"
    )

    if limit_n is not None:
        df = df.iloc[:limit_n, :]

    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    return df.to_numpy(dtype=float)


def standardize_like_R(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=1, keepdims=True)
    sigma[sigma == 0.0] = 1.0

    return (X - mu) / sigma


def pairwise_euclidean(X):
    a = np.sum(X * X, axis=1, keepdims=True)
    D2 = a + a.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2, dtype=float)
    np.fill_diagonal(D, 0.0)

    return D


def solve_kmedoids_ilp(
    D,
    k,
    time_limit,
    mip_gap,
    threads,
    seed,
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
    m.Params.OutputFlag = 1

    # variáveis
    y = m.addVars(n, vtype=GRB.BINARY, name="y")
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

    # objetivo: soma total das distâncias (custo médio = objetivo / n)
    m.setObjective(
        quicksum(D[i, j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE
    )

    # cada ponto j é atendido por exatamente um medoid
    for j in range(n):
        m.addConstr(quicksum(x[i, j] for i in range(n)) == 1, name=f"assign[{j}]")

    # link: só pode atribuir a i se i for medoid
    for i in range(n):
        for j in range(n):
            m.addConstr(x[i, j] <= y[i], name=f"link[{i},{j}]")

    # exatamente k medoids
    m.addConstr(quicksum(y[i] for i in range(n)) == k, name="kmedoids")

    # força medoid i a atender-se: evita soluções degeneradas e acelera
    for i in range(n):
        m.addConstr(x[i, i] == y[i], name=f"self[{i}]")

    # otimizar
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
    return status, best_obj, y_sol, x_sol, runtime, mipgap_out


def write_outputs(out_prefix, D, k, status, obj, y_sol, x_sol, runtime, mipgap):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pref = str(out_prefix)

    n = D.shape[0]
    avg_obj = obj / n if n > 0 and math.isfinite(obj) else float("nan")

    medoids = np.where(y_sol == 1)[0] + 1
    pd.Series(medoids, name="medoid_index_1based").to_csv(
        f"{pref}_medoids.csv", index=False
    )

    assign = np.argmax(x_sol, axis=0)
    labels = assign + 1
    pd.DataFrame({"point": np.arange(1, n + 1, dtype=int), "cluster": labels}).to_csv(
        f"{pref}_clustering.csv", index=False
    )

    with open(f"{pref}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"status={status}\n")
        f.write(f"objective_total={obj}\n")
        f.write(f"objective_avg_per_point={avg_obj}\n")
        f.write(f"k={k}\n")
        f.write(f"n={n}\n")
        f.write(f"runtime_sec={runtime}\n")
        if mipgap is not None and not (
            isinstance(mipgap, float) and math.isnan(mipgap)
        ):
            f.write(f"mip_gap={mipgap}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        required=True,
        help="Caminho do arquivo (ex.: Data_Sets_50/haberman.i)",
    )
    ap.add_argument("--k", type=int, required=True, help="Número de clusters/medoids")
    ap.add_argument(
        "--scale", action="store_true", help="Padronizar colunas (R::scale, ddof=1)"
    )
    ap.add_argument("--sep", default=";", help="Separador (default ';')")
    ap.add_argument("--decimal", default=",", help="Decimal (default ',')")

    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--mip-gap", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument(
        "--out-prefix",
        default="results/kmedoids_ilp",
        help="Prefixo de saída (sem extensão)",
    )
    args = ap.parse_args()

    X = load_points(Path(args.data), sep=args.sep, decimal=args.decimal, limit_n=None)
    if args.scale:
        X = standardize_like_R(X)
    D = pairwise_euclidean(X)

    status, obj, y_sol, x_sol, runtime, mipgap = solve_kmedoids_ilp(
        D,
        args.k,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
    )
    write_outputs(
        Path(args.out_prefix), D, args.k, status, obj, y_sol, x_sol, runtime, mipgap
    )

    n = D.shape[0]
    print(
        f"Done. status={status}, objective_total={obj:.6f}, objective_avg={obj/n:.6f}, time={runtime:.2f}s"
    )
    print(f"Outputs at: {args.out_prefix}*")


if __name__ == "__main__":
    main()
