import numpy as np
import os
import matplotlib.pyplot as plt

def plot_amount_of_wins(results, title="Clustering Results"):
    """
    For every pair (file, k) in results, find the config with best max_value and keep count of how many wins.
    The plot should show the amount of best results per method.
    the results are organized like:
    config,file,n,k,alpha,construct_mode,ls_mode,reactive_alphas,reactive_block,sample_size,iterations,time_limit_s,timed_out,max_value,size,feasible,time_s,elements
    """
    method_wins = {}
    instances_results = {}
    for result in results:
        method = result[0]
        max_value = -float(result[13])
        instance = (result[1], result[3])  # (file, k)
        if instance not in instances_results:
            instances_results[instance] = []
        instances_results[instance].append((method, max_value))

    for instance, method_values in instances_results.items():
        best_value = np.inf
        for method, value in method_values:
            if value < best_value:
                best_value = value
        for method, value in method_values:
            if value == best_value:
                if method not in method_wins:
                    method_wins[method] = 0
                method_wins[method] += 1

    methods = list(method_wins.keys())
    wins = [method_wins[method] for method in methods]

    plt.bar(methods, wins)
    plt.xlabel('Methods')
    plt.ylabel('Number of Best Results')
    plt.title(title)
    plt.savefig("plots/clustering_results.png")
    plt.close()
    return method_wins

def performance_profile_normalized(results):
    normalized_objectives = {}
    instances_results = {}
    for result in results:
        method = result[0]
        max_value = float(result[13])
        max_value = -max_value
        time = float(result[16])
        instance = (result[1], result[3])  # (file, k, time)
        if instance not in instances_results:
            instances_results[instance] = []
        instances_results[instance].append((method, max_value, time))

    for instance, method_values in instances_results.items():
        best_value = np.inf
        for method, value, _ in method_values:
            if value < best_value:
                best_value = value
        for method, value, time in method_values:
            normalized_value = (2 * ( best_value / value) - 1.74)/0.26 if value != 0 else 0
            
            if method not in normalized_objectives:
                normalized_objectives[method] = []
            normalized_objectives[method].append((normalized_value, time))
    
    worst_performance = {}
    for method in normalized_objectives.keys():
        worst = min(v[0] for v in normalized_objectives[method])
        worst_performance[method] = worst

    print("Worst performances per method:")
    for method, worst in worst_performance.items():
        print(f"{method}: {worst}")


    normalized_sum = {}
    for method, values in normalized_objectives.items():
        total = sum(v[0] for v in values)
        normalized_sum[method] = total / len(values)
    
    # plot normalized_sum

    methods = list(normalized_sum.keys())
    norms = [normalized_sum[method] for method in methods]
    plt.bar(methods, norms)
    plt.xlabel('Methods')
    plt.ylabel('Average Normalized Objective')
    plt.title('Performance Profile - Normalized Objectives')
    plt.savefig("plots/scores.png")
    plt.close()

    return normalized_objectives
    
def comparison_time(results):
    accumulated = {}
    
    #sort each method's results by time
    for method in results.keys():
        results[method] = sorted(results[method], key=lambda x: x[1])

    for method in results.keys():
        accumulated[method] = ([],[])  # (times, summed_normalized objectives)
        for norm_obj, time in results[method]:
            accumulated[method][0].append(time)
            if len(accumulated[method][1]) == 0:
                accumulated[method][1].append(norm_obj)
            else:
                accumulated[method][1].append(accumulated[method][1][-1] + norm_obj)
    
    for method in accumulated.keys():
        times, sums = accumulated[method]
        sums = [s / len(results[method]) for s in sums]  # normalize by number of instances
        plt.loglog(times, sums, label=method)
    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated Normalized Objective')
    plt.title('Comparison over Time')
    plt.legend()
    plt.savefig("plots/comparison_time.png")
    plt.close()





if __name__ == "__main__":
    # open csv file and read results
    results = []
    with open("results/grasp_kmedoids_20251112_001656.csv", 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            results.append(parts)
    plot_amount_of_wins(results)
    results = performance_profile_normalized(results)
    comparison_time(results)

