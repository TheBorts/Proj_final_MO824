#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "metaheuristics/grasp/AbstractGRASP.h"
#include "problems/kmedoids/common.h"
#include "problems/kmedoids/solvers/GRASP_KMedoids.h"
#include "problems/kmedoids/solvers/GRASP_KMedoids_RPG.h"

using namespace std;

string INSTANCES_DIR = "instances/general";

vector<string> INSTANCE_FILES = {
    "haberman.i",
};

bool USE_ZSCORE = true;

vector<int> K_VALUES = {3};

vector<double> ALPHA_VALUES = {0.05, 0.90};
vector<int> P_VALUES = {5, 10, 20};

long MAX_TIME_MILLIS = 30L * 60L * 1000L;  // 30 min
int MAX_TOTAL_ITERATIONS = 1000;

string RESULTS_DIR = "src_/problems/kmedoids/solvers/results";

static void silence_framework() { AbstractGRASP<int>::verbose = false; }

enum class SolverKind
{
    Standard,
    RPG
};

struct ExperimentConfig
{
    string name;
    SolverKind kind;
    double alpha;
    int p;
};

struct ExperimentResult
{
    string config_name;
    string instance_file;
    int k{};
    double best_total_cost{};
    double best_avg_cost{};
    int total_iterations{};
    int iterations_to_best{};
    long execution_time_ms{};
    bool stopped_by_time{false};
    int solution_size{};
};

class GRASP_KMedoids_WithStopping : public GRASP_KMedoids
{
   public:
    GRASP_KMedoids_WithStopping(double alpha, int iterations, vector<vector<double>>& D,
                                int k, long max_time_ms)
        : GRASP_KMedoids(alpha, iterations, D, k), max_time_ms_(max_time_ms)
    {
    }

    Solution<int> solve()
    {
        bestSol = createEmptySol();
        auto t0 = chrono::steady_clock::now();
        int i = 0;

        while (i < iterations)
        {
            auto now = chrono::steady_clock::now();
            auto ms = chrono::duration_cast<chrono::milliseconds>(now - t0).count();
            if (ms > max_time_ms_)
            {
                stopped_by_time = true;
                break;
            }

            constructiveHeuristic();
            localSearch();

            if (bestSol->cost > sol->cost)
            {
                bestSol = sol;
                iterations_to_best = i;
            }
            ++i;
        }

        total_iterations = i;
        auto tf = chrono::steady_clock::now();
        execution_time_ms =
            static_cast<long>(chrono::duration_cast<chrono::milliseconds>(tf - t0).count());
        return *bestSol;
    }

    int total_iterations{0};
    int iterations_to_best{0};
    long execution_time_ms{0};
    bool stopped_by_time{false};

   private:
    long max_time_ms_;
};

class GRASP_KMedoids_RPG_WithStopping : public GRASP_KMedoids_RPG
{
   public:
    GRASP_KMedoids_RPG_WithStopping(double alpha, int iterations, vector<vector<double>>& D,
                                    int k, int p, long max_time_ms)
        : GRASP_KMedoids_RPG(alpha, iterations, D, k, p), max_time_ms_(max_time_ms)
    {
    }

    Solution<int> solve()
    {
        bestSol = createEmptySol();
        auto t0 = chrono::steady_clock::now();
        int i = 0;

        while (i < iterations)
        {
            auto now = chrono::steady_clock::now();
            auto ms = chrono::duration_cast<chrono::milliseconds>(now - t0).count();
            if (ms > max_time_ms_)
            {
                stopped_by_time = true;
                break;
            }

            constructiveHeuristic();
            localSearch();

            if (bestSol->cost > sol->cost)
            {
                bestSol = sol;
                iterations_to_best = i;
            }
            ++i;
        }

        total_iterations = i;
        auto tf = chrono::steady_clock::now();
        execution_time_ms =
            static_cast<long>(chrono::duration_cast<chrono::milliseconds>(tf - t0).count());
        return *bestSol;
    }

    int total_iterations{0};
    int iterations_to_best{0};
    long execution_time_ms{0};
    bool stopped_by_time{false};

   private:
    long max_time_ms_;
};

static vector<ExperimentConfig> generate_configurations()
{
    vector<ExperimentConfig> cfgs;
    cfgs.reserve(ALPHA_VALUES.size() + P_VALUES.size());

    for (double a : ALPHA_VALUES)
    {
        cfgs.push_back(
            ExperimentConfig{"GRASP_alpha=" + to_string(a), SolverKind::Standard, a, -1});
    }
    for (int p : P_VALUES)
    {
        cfgs.push_back(ExperimentConfig{"RPG_p=" + to_string(p), SolverKind::RPG, -1.0, p});
    }
    return cfgs;
}

static vector<vector<double>> load_distance_matrix(string& instance_path)
{
    auto X = load_i_dataset(instance_path, ';', ',');
    if (USE_ZSCORE)
    {
        zscore_inplace(X, 1);
    }
    return pairwise_euclidean(X);
}

static ExperimentResult run_experiment(ExperimentConfig& config, string& instance_file, int k)
{
    cout << "  Running: " << config.name << " | k=" << k << " | on " << instance_file << "\n";

    string instance_path = INSTANCES_DIR + "/" + instance_file;
    auto D = load_distance_matrix(instance_path);
    int n = static_cast<int>(D.size());

    Solution<int> sol;
    int total_iterations = 0, iterations_to_best = 0;
    long exec_ms = 0;
    bool stopped_by_time = false;

    if (config.kind == SolverKind::Standard)
    {
        GRASP_KMedoids_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                          MAX_TIME_MILLIS);
        sol = grasp.solve();
        total_iterations = grasp.total_iterations;
        iterations_to_best = grasp.iterations_to_best;
        exec_ms = grasp.execution_time_ms;
        stopped_by_time = grasp.stopped_by_time;
    }
    else
    {  // RPG
        GRASP_KMedoids_RPG_WithStopping grasp(0.0, MAX_TOTAL_ITERATIONS, D, k, config.p,
                                              MAX_TIME_MILLIS);
        sol = grasp.solve();
        total_iterations = grasp.total_iterations;
        iterations_to_best = grasp.iterations_to_best;
        exec_ms = grasp.execution_time_ms;
        stopped_by_time = grasp.stopped_by_time;
    }

    double best_avg = sol.cost;  // média por ponto
    double best_total = best_avg * static_cast<double>(n);

    cout << "    -> Total: " << fixed << setprecision(6) << best_total << " | Avg: " << best_avg
         << ", Iter: " << total_iterations << ", Time: " << setprecision(3)
         << (static_cast<double>(exec_ms) / 1000.0) << "s\n";

    ExperimentResult r;
    r.config_name = config.name;
    r.instance_file = instance_file;
    r.k = k;
    r.best_total_cost = best_total;
    r.best_avg_cost = best_avg;
    r.total_iterations = total_iterations;
    r.iterations_to_best = iterations_to_best;
    r.execution_time_ms = exec_ms;
    r.stopped_by_time = stopped_by_time;
    r.solution_size = static_cast<int>(sol.size());
    return r;
}

static void save_results_to_csv(vector<ExperimentResult>& results, string& output_file)
{
    filesystem::create_directories(filesystem::path(output_file).parent_path());

    ofstream f(output_file, ios::out | ios::trunc);
    if (!f.is_open())
    {
        throw runtime_error("Failed to open output CSV: " + output_file);
    }

    f << "Configuration,Instance,k,TotalCost,AvgCost,TotalIterations,"
         "IterationsToBest,ExecutionTimeMs,StoppedByTime,SolutionSize\n";

    f.setf(ios::fixed);
    for (auto& r : results)
    {
        f << r.config_name << ',' << r.instance_file << ',' << r.k << ',' << setprecision(6)
          << r.best_total_cost << ',' << setprecision(6) << r.best_avg_cost << ','
          << r.total_iterations << ',' << r.iterations_to_best << ',' << r.execution_time_ms << ','
          << (r.stopped_by_time ? "true" : "false") << ',' << r.solution_size << '\n';
    }
    cout << "\nResults saved to: " << output_file << "\n";
}

static void print_summary_statistics(vector<ExperimentResult>& results)
{
    cout << "\n=== SUMMARY STATISTICS ===\n\n";

    vector<pair<string, int>> keys;
    for (auto& r : results) keys.emplace_back(r.instance_file, r.k);
    sort(keys.begin(), keys.end());
    keys.erase(unique(keys.begin(), keys.end()), keys.end());

    for (auto [inst, k] : keys)
    {
        vector<ExperimentResult> subset;
        for (auto& r : results)
            if (r.instance_file == inst && r.k == k) subset.push_back(r);
        if (subset.empty()) continue;

        auto best_it = min_element(subset.begin(), subset.end(),
                                   [](ExperimentResult& a, ExperimentResult& b)
                                   { return a.best_avg_cost < b.best_avg_cost; });

        auto& best = *best_it;
        cout << "Instance: " << inst << " | k=" << k << "\n";
        cout << "  Best configuration: " << best.config_name << "\n";
        cout << "  Best avg cost: " << fixed << setprecision(6) << best.best_avg_cost << "\n";
        cout << "  Total cost: " << best.best_total_cost << "\n";
        cout << "  Iterations: " << best.total_iterations << "\n";
        cout << "  Time: " << setprecision(3)
             << (static_cast<double>(best.execution_time_ms) / 1000.0) << " seconds\n\n";
    }
}

int main()
{
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    tm tm{};

    #ifdef _WIN32
        localtime_s(&tm, &t);
    #else
        localtime_r(&t, &tm);
    #endif

    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

    string output_csv = RESULTS_DIR + "/grasp_kmedoids_" + string(buf) + ".csv";

    cout << "==============================================\n";
    cout << "GRASP K-MEDOIDS\n";
    cout << "==============================================\n";
    cout << "Tempo máximo por rodada: " << (MAX_TIME_MILLIS / 1000) << " seconds\n";
    cout << "Arquivo de saída: " << output_csv << "\n";
    cout << "==============================================\n\n";

    auto configs = generate_configurations();
    vector<ExperimentResult> all_results;

    cout << "Total configurações: " << configs.size() << "\n";
    cout << "Total instâncias: " << INSTANCE_FILES.size() << "\n";
    cout << "Total experimentos: " << (configs.size() * INSTANCE_FILES.size() * K_VALUES.size())
         << "\n\n";
    cout << "Start...\n\n";

    auto start = chrono::steady_clock::now();
    int exp_count = 0;

    for (auto& inst : INSTANCE_FILES)
    {
        cout << "\n--- Instância: " << inst << " ---\n";
        for (int k : K_VALUES)
        {
            for (auto& cfg : configs)
            {
                ++exp_count;
                cout << "\n[Experimento " << exp_count << "/"
                     << (configs.size() * INSTANCE_FILES.size() * K_VALUES.size()) << "]\n";
                auto res = run_experiment(cfg, inst, k);
                all_results.push_back(res);

                if (exp_count % 10 == 0)
                {
                    save_results_to_csv(all_results, output_csv);
                }
            }
        }
    }

    save_results_to_csv(all_results, output_csv);

    auto end = chrono::steady_clock::now();
    double total_time_min = chrono::duration_cast<chrono::seconds>(end - start).count() / 60.0;

    cout << "Experimentos completos\n";
    cout << "Tempo Total: " << fixed << setprecision(2) << total_time_min << " minutos\n";
    cout << "Resultados salvos: " << output_csv << "\n";

    print_summary_statistics(all_results);
    return 0;
}
