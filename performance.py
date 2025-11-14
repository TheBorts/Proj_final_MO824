import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl

from performance_profile import _norm_file

DATA_PATH_GRASP = "results/grasp_kmedoids_merged.csv"
DATA_PATH_HEURISTICS = "src/literature_heuristics/results/clustering_results_fixed.csv"


def carregar_resultados(caminho):
    df = pd.read_csv(caminho)

    # Add an index column and identify each row
    df.insert(0, 'index', range(len(df)))

    df["config"] = df["config"].astype(str)
    df["file"] = df["file"].astype(str).map(_norm_file)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")
    df["max_value"] = -pd.to_numeric(df["max_value"], errors="coerce")

    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    else:
        df["time_s"] = np.nan

    df["instancia"] = df["file"].astype(str) + "|k=" + df["k"].astype(str)

    return df


def carregar_resultados_heuristicas(caminho):
    df = pd.read_csv(caminho)

    # Add an index column and identify each row
    df.insert(0, 'index', range(len(df)))

    df["config"] = df["Method"].astype(str)
    df["file"] = df["Instance"].astype(str).map(_norm_file)
    df["k"] = pd.to_numeric(df["Num_Clusters"], errors="coerce").astype("Int64")
    df["max_value"] = pd.to_numeric(df["Cost_Per_Point"], errors="coerce")

    if "Execution_Time" in df.columns:
        df["time_s"] = pd.to_numeric(df["Execution_Time"], errors="coerce")*1000.0
        df["time_to_solution_s"] = pd.to_numeric(df["Execution_Time"], errors="coerce")*1000.0
    else:
        df["time_s"] = np.nan
        df["time_to_solution_s"] = np.nan

    df["instancia"] = df["file"].astype(str) + "|k=" + df["k"].astype(str)

    return df


def find_best_solutions_per_instance(df):
    # Group by 'instancia' and 'k'
    grouped = df.groupby(['instancia', 'k'])
    best_values = grouped['max_value'].min().reset_index()

    # Find the rows in the original dataframe that correspond to these best values and instances
    best_solutions = pd.merge(best_values, df, on=['instancia', 'k', 'max_value'], how='left')    

    return best_solutions


def find_best_solutions_per_instance_heuristics(df):
    # Group by 'instancia' and 'Num_Clusters'
    grouped = df.groupby(['instancia', 'Num_Clusters'])
    best_values = grouped['Cost'].min().reset_index()

    # Find the rows in the original dataframe that correspond to these best values and instances
    best_solutions = pd.merge(best_values, df, on=['instancia', 'Num_Clusters', 'Cost'], how='left')    

    return best_solutions


def build_performance_profile(best_solutions, df):
    
    solved_counts = {config: [] for config in df['config'].unique()}    

    # For every config in the dataframe, plot the performance profile
    configs = df['config'].unique()
    ammount_of_runs_per_config = {config: len(df[df['config'] == config]) for config in configs}

    for config in configs:
        
        config_df = df[df['config'] == config]

        # if row index in best_solutions is in config_df, count as solved
        for _, row in best_solutions.iterrows():
            if row['config'] == config:
                solved_counts[config].append(row["time_to_solution_s"])
            else:
                continue
            
    # sort solved_counts by time
    for config in solved_counts:
        solved_counts[config].sort()
    
    performance_values = {}
    for config in solved_counts:
        performance_values[config] = [ (i + 1) / ammount_of_runs_per_config[config] for i in range(len(solved_counts[config])) ]
        #print(f"Config: {config} : last_time = {solved_counts[config][-1]}")

    #for config in solved_counts:
        #solved_counts[config].append(432628.0)
        #performance_values[config].append(performance_values[config][-1])

    return solved_counts, performance_values


def build_performance_profile_heuristics(best_solutions, df):
    
    solved_counts = {config: [] for config in df['Method'].unique()}    

    # For every config in the dataframe, plot the performance profile
    configs = df['Method'].unique()
    ammount_of_runs_per_config = {config: len(df[df['Method'] == config]) for config in configs}

    for config in configs:
        
        config_df = df[df['Method'] == config]

        # if row index in best_solutions is in config_df, count as solved
        for _, row in best_solutions.iterrows():
            if row['index'] in config_df['index'].values:
                solved_counts[config].append(row["Execution_Time"])
            else:
                continue
            
    # sort solved_counts by time
    for config in solved_counts:
        solved_counts[config].sort()
    
    performance_values = {}
    for config in solved_counts:
        performance_values[config] = [ (i + 1) / ammount_of_runs_per_config[config] for i in range(len(solved_counts[config])) ]
        #print(f"Config: {config} : last_time = {solved_counts[config][-1]}")

    for config in solved_counts:
        solved_counts[config].append(2.5)
        performance_values[config].append(performance_values[config][-1])

    return solved_counts, performance_values


def plot_performance_profile(solved_counts, performance_values, output_path):

    # Use a nicer style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Make fonts larger and consistent
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['legend.fontsize'] = 10

    # Line + marker combinations (repeat if more configs exist)
    line_styles = ["-", "--", "-.", ":", "-"]
    markers = ["o", "s", "D", "^", "v"]
    
    plt.figure(figsize=(8, 6))

    for i, config in enumerate(solved_counts):
        style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        plt.plot(
            solved_counts[config], 
            performance_values[config],
            style,
            marker=marker,
            markersize=5,
            linewidth=2,
            label=config
        )

    plt.title("Performance Profile")
    plt.xlabel("Time passed (s)")
    plt.ylabel("Proportion of Problems Solved")

    #plt.xlim(left=0, right=2.5)
    #plt.ylim(bottom=0, top=1.05)

    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    # Ensure nothing is clipped
    plt.tight_layout()

    # Save with high resolution
    plt.savefig(output_path, dpi=300)
    plt.close()

DISCREETNESS = 10000

def prepare_data_for_leftover_analysis(df, best_solutions):
    solutions_per_gap = {config: [0]*DISCREETNESS for config in df['config'].unique()}
    configs = df['config'].unique()

    ammount_of_runs_per_config = {config: len(df[df['config'] == config]) for config in configs}

    # Get best known values per pair (instance, k)
    best_known = {}
    for _, row in best_solutions.iterrows():
        key = (row['instancia'], row['k'])
        best_known[key] = row['max_value']

    
    for config in configs:
        config_df = df[df['config'] == config]

        for _, row in config_df.iterrows():
            key = (row['instancia'], row['k'])
            if key in best_known:
                best_value = best_known[key]
                gap = (row['max_value'] - best_value) / abs(best_value) if best_value != 0 else 0.0
                gap_percentage = int(np.ceil(gap * DISCREETNESS))
                #print(f"real_gap: {gap}, gap_percentage: {gap_percentage}")
                if 0 <= gap_percentage < DISCREETNESS:
                    solutions_per_gap[config][gap_percentage] += 1 / ammount_of_runs_per_config[config]

    # make cumulative
    for config in solutions_per_gap:
        for i in range(1, DISCREETNESS):
            solutions_per_gap[config][i] += solutions_per_gap[config][i - 1]

    return solutions_per_gap


def prepare_data_for_leftover_analysis_heuristics(df, best_solutions):
    solutions_per_gap = {config: [0]*DISCREETNESS for config in df['Method'].unique()}
    configs = df['Method'].unique()

    ammount_of_runs_per_config = {config: len(df[df['Method'] == config]) for config in configs}

    # Get best known values per pair (instance, Num_Clusters)
    best_known = {}
    for _, row in best_solutions.iterrows():
        key = (row['instancia'], row['Num_Clusters'])
        best_known[key] = row['Cost']

    
    for config in configs:
        config_df = df[df['Method'] == config]

        for _, row in config_df.iterrows():
            key = (row['instancia'], row['Num_Clusters'])
            if key in best_known:
                best_value = best_known[key]
                gap = (row['Cost'] - best_value) / abs(best_value) if best_value != 0 else 0.0
                gap_percentage = int(np.ceil(gap * DISCREETNESS))
                #print(f"real_gap: {gap}, gap_percentage: {gap_percentage}")
                if 0 <= gap_percentage < DISCREETNESS:
                    solutions_per_gap[config][gap_percentage] += 1 / ammount_of_runs_per_config[config]

    # make cumulative
    for config in solutions_per_gap:
        for i in range(1, DISCREETNESS):
            solutions_per_gap[config][i] += solutions_per_gap[config][i - 1]

    return solutions_per_gap    


PLOT_PERCENTAGE = DISCREETNESS * 13 / 100


def plot_leftover_solutions(solutions_per_gap, output_path):
    plt.style.use("seaborn-v0_8-darkgrid")

    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['legend.fontsize'] = 10

    line_styles = ["-", "--", "-.", ":", "-"]
    markers = ["o", "s", "D", "^", "v"]
    
    plt.figure(figsize=(8, 6))

    x_values = [x / DISCREETNESS * 100 for x in range(DISCREETNESS)]  # convert to percentage

    for i, config in enumerate(solutions_per_gap):
        style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        plt.plot(
            x_values[: int(PLOT_PERCENTAGE)], 
            solutions_per_gap[config][: int(PLOT_PERCENTAGE)],
            style,
            marker=marker,
            markersize=5,
            linewidth=2,
            label=config
        )

    plt.title("Leftover Solutions vs. Gap Percentage")
    plt.xlabel("Gap Percentage (%)")
    plt.ylabel("Proportion of Problems Solved")

    #plt.ylim(bottom=0, top=1.05)

    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_grasps_performance():
    df = carregar_resultados(DATA_PATH_GRASP)
    best_solutions = find_best_solutions_per_instance(df)
    solved_counts, performance_values = build_performance_profile(best_solutions, df)
    plot_performance_profile(solved_counts, performance_values, "plots_performance/performance_profile.png")
    print("Performance profile saved to plots_performance/performance_profile.png")
    solutions_per_gap = prepare_data_for_leftover_analysis(df, best_solutions)
    plot_leftover_solutions(solutions_per_gap, "plots_performance/leftover_solutions.png")
    print("Leftover solutions plot saved to plots_performance/leftover_solutions.png")

def plot_heuristics_performance():
    df = carregar_resultados_heuristicas(DATA_PATH_HEURISTICS)
    best_solutions = find_best_solutions_per_instance_heuristics(df)
    solved_counts, performance_values = build_performance_profile_heuristics(best_solutions, df)
    plot_performance_profile(solved_counts, performance_values, "plots_performance/performance_profile_heuristics.png")
    print("Performance profile saved to plots_performance/performance_profile_heuristics.png")
    solutions_per_gap = prepare_data_for_leftover_analysis_heuristics(df, best_solutions)
    plot_leftover_solutions(solutions_per_gap, "plots_performance/leftover_solutions_heuristics.png")
    print("Leftover solutions plot saved to plots_performance/leftover_solutions_heuristics.png")


def combine_best_solutions(best_heuristics, best_grasp):
    """
        Combines the best solutions from heuristics and GRASP, selecting the overall best for each (instance, k) pair.
        Problem is that the columns have different names for the solution values and instance characteristics.
        Returns a DataFrame with the combined best solutions.
    """
    
    combined_rows = []

    # Create a mapping for heuristics best solutions
    heuristics_map = {}
    for _, row in best_heuristics.iterrows():
        key = (row['instancia'], row['Num_Clusters'])
        heuristics_map[key] = row

    for _, row in best_grasp.iterrows():
        key = (row['instancia'], row['k'])
        if key in heuristics_map:
            heuristic_row = heuristics_map[key]
            if row['max_value'] < heuristic_row['Cost']:
                combined_rows.append(row)
            else:
                combined_rows.append(heuristic_row)
        else:
            combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows)
    return combined_df


def combine_solved_counts_and_performance(solved_counts_heuristics, performance_values_heuristics,
                                          solved_counts_grasp, performance_values_grasp):
    combined_solved_counts = {}
    combined_performance_values = {}

    for config in solved_counts_heuristics:
        if config == "fasterpam":
            combined_solved_counts[config] = solved_counts_heuristics[config]
            combined_performance_values[config] = performance_values_heuristics[config]
    
    for config in solved_counts_grasp:
        if config == "GRASP_FI_alpha=0.050000":
            combined_solved_counts[config] = solved_counts_grasp[config]
            combined_performance_values[config] = performance_values_grasp[config]

    return combined_solved_counts, combined_performance_values



def compare_heuristics_grasp():
    df_grasp = carregar_resultados(DATA_PATH_GRASP)
    df_heuristics = carregar_resultados_heuristicas(DATA_PATH_HEURISTICS)
    
    best_solution_grasp = find_best_solutions_per_instance(df_grasp)
    best_solution_heuristics = find_best_solutions_per_instance(df_heuristics)

    combined_best_solutions = combine_best_solutions(best_solution_heuristics, best_solution_grasp)

    solved_counts_grasp, performance_values_grasp = build_performance_profile(combined_best_solutions, df_grasp)
    solved_counts_heuristics, performance_values_heuristics = build_performance_profile(combined_best_solutions, df_heuristics)

    combined_solved_counts, combined_performance_values = combine_solved_counts_and_performance(
        solved_counts_heuristics, performance_values_heuristics,
        solved_counts_grasp, performance_values_grasp
    )

    plot_performance_profile(combined_solved_counts, combined_performance_values, "plots_performance/performance_profile_combined.png")
    print("Combined performance profile saved to plots_performance/performance_profile_combined.png")


if __name__ == "__main__":
    compare_heuristics_grasp()