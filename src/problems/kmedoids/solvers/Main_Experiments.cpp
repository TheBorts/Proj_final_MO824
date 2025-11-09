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

long MAX_TIME_MILLIS = 1 * 60L * 1000L;
int MAX_TOTAL_ITERATIONS = 1000;

string RESULTS_DIR = "results";

bool ENABLE_TTT_MODE = false;
int TTT_RUNS = 50;

vector<tuple<string, int, double>> TTT_TARGETS = {make_tuple("haberman.i", 3, 1.127419)};

double get_target_avg_for(string& file, int k)
{
    for (auto& t : TTT_TARGETS)
        if (get<0>(t) == file && get<1>(t) == k) return get<2>(t);
    return -1.0;
}

void silence_framework() { AbstractGRASP<int>::verbose = false; }

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
    string config;
    string file;
    int n{};
    int k{};
    double alpha{};
    string construct_mode;
    string ls_mode;
    string reactive_alphas;
    string reactive_block;
    string sample_size;
    int iterations{};
    double time_limit_s{};
    bool timed_out{};
    double max_value{};
    int size{};
    bool feasible{};
    double time_s{};
    string elements;
};

class GRASP_KMedoids_WithStopping : public GRASP_KMedoids
{
   public:
    GRASP_KMedoids_WithStopping(double alpha, int iterations, vector<vector<double>>& D, int k,
                                long max_time_ms, double target_avg_value = -1.0,
                                bool ttt_mode = false)
        : GRASP_KMedoids(alpha, iterations, D, k),
          max_time_ms_(max_time_ms),
          target_avg_value_(target_avg_value),
          ttt_mode_(ttt_mode)
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

                if (ttt_mode_ && target_avg_value_ > 0.0 && time_to_target_ms < 0)
                {
                    if (bestSol->cost <= target_avg_value_)
                    {
                        time_to_target_ms = static_cast<long>(ms);
                        break;
                    }
                }
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
    long time_to_target_ms{-1};

   private:
    long max_time_ms_;
    double target_avg_value_;
    bool ttt_mode_;
};

class GRASP_KMedoids_RPG_WithStopping : public GRASP_KMedoids_RPG
{
   public:
    GRASP_KMedoids_RPG_WithStopping(double alpha, int iterations, vector<vector<double>>& D, int k,
                                    int p, long max_time_ms, double target_avg_value = -1.0,
                                    bool ttt_mode = false)
        : GRASP_KMedoids_RPG(alpha, iterations, D, k, p),
          max_time_ms_(max_time_ms),
          target_avg_value_(target_avg_value),
          ttt_mode_(ttt_mode)
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

                if (ttt_mode_ && target_avg_value_ > 0.0 && time_to_target_ms < 0)
                {
                    if (bestSol->cost <= target_avg_value_)
                    {
                        time_to_target_ms = static_cast<long>(ms);
                        break;
                    }
                }
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
    long time_to_target_ms{-1};

   private:
    long max_time_ms_;
    double target_avg_value_;
    bool ttt_mode_;
};

vector<ExperimentConfig> generate_configurations()
{
    vector<ExperimentConfig> cfgs;
    cfgs.reserve(ALPHA_VALUES.size() + P_VALUES.size());

    for (double a : ALPHA_VALUES)
        cfgs.push_back(
            ExperimentConfig{"GRASP_alpha=" + to_string(a), SolverKind::Standard, a, -1});

    for (int p : P_VALUES)
        cfgs.push_back(ExperimentConfig{"RPG_p=" + to_string(p), SolverKind::RPG, -1.0, p});

    return cfgs;
}

vector<vector<double>> load_distance_matrix(string& instance_path)
{
    auto X = load_i_dataset(instance_path, ';', ',');
    if (USE_ZSCORE) zscore_inplace(X, 1);
    return pairwise_euclidean(X);
}

void save_ttt_header_if_needed(string& csv_path)
{
    filesystem::create_directories(filesystem::path(csv_path).parent_path());
    bool exists = filesystem::exists(csv_path);
    ofstream f(csv_path, ios::out | ios::app);
    if (!f.is_open()) throw runtime_error("Failed to open TTT CSV: " + csv_path);
    if (!exists) f << "instance,file,k,config,target_avg,run_idx,time_to_target_ms\n";
}

void append_ttt_line(string& csv_path, string& instance_file, int k, string& config_name,
                     double target_avg, int run_idx, long ttt_ms)
{
    ofstream f(csv_path, ios::out | ios::app);
    if (!f.is_open()) throw runtime_error("Failed to open TTT CSV: " + csv_path);
    f.setf(ios::fixed);
    f << INSTANCES_DIR << '/' << instance_file << ',' << instance_file << ',' << k << ','
      << config_name << ',' << setprecision(6) << target_avg << ',' << run_idx << ',' << ttt_ms
      << '\n';
}

ExperimentResult run_experiment(ExperimentConfig& config, string& instance_file, int k,
                                string& ttt_csv)
{
    cout << "  Running: " << config.name << " | k=" << k << " | on " << instance_file << "\n";

    string instance_path = INSTANCES_DIR + "/" + instance_file;
    auto D = load_distance_matrix(instance_path);
    int n = static_cast<int>(D.size());

    if (ENABLE_TTT_MODE)
    {
        double target_avg = get_target_avg_for(instance_file, k);
        if (!(target_avg > 0.0))
        {
            cout << "    [WARN] target_avg não definido (>0). Os tempos ficarão como -1.\n";
        }

        save_ttt_header_if_needed(ttt_csv);

        for (int run = 0; run < TTT_RUNS; ++run)
        {
            if (config.kind == SolverKind::Standard)
            {
                GRASP_KMedoids_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                                  MAX_TIME_MILLIS, target_avg, true);
                auto sol = grasp.solve();
                append_ttt_line(ttt_csv, instance_file, k, config.name, target_avg, run,
                                grasp.time_to_target_ms);
            }
            else
            {
                GRASP_KMedoids_RPG_WithStopping grasp(0.0, MAX_TOTAL_ITERATIONS, D, k, config.p,
                                                      MAX_TIME_MILLIS, target_avg, true);
                auto sol = grasp.solve();
                append_ttt_line(ttt_csv, instance_file, k, config.name, target_avg, run,
                                grasp.time_to_target_ms);
            }
        }

        ExperimentResult r{};
        r.config = config.name;
        r.file = instance_file;
        r.n = n;
        r.k = k;
        r.alpha = (config.kind == SolverKind::Standard) ? config.alpha : -1.0;
        r.construct_mode = (config.kind == SolverKind::Standard) ? "STANDARD" : "RPG";
        r.ls_mode = "BEST_IMPROVING";
        r.iterations = MAX_TOTAL_ITERATIONS;
        r.time_limit_s = static_cast<double>(MAX_TIME_MILLIS) / 1000.0;
        r.timed_out = false;
        r.max_value = 0.0;
        r.size = 0;
        r.feasible = true;
        r.time_s = 0.0;
        r.elements = "";
        return r;
    }

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
    {
        GRASP_KMedoids_RPG_WithStopping grasp(0.0, MAX_TOTAL_ITERATIONS, D, k, config.p,
                                              MAX_TIME_MILLIS);
        sol = grasp.solve();
        total_iterations = grasp.total_iterations;
        iterations_to_best = grasp.iterations_to_best;
        exec_ms = grasp.execution_time_ms;
        stopped_by_time = grasp.stopped_by_time;
    }

    double best_avg = sol.cost;
    double best_total = best_avg * static_cast<double>(n);
    double time_sec = static_cast<double>(exec_ms) / 1000.0;

    cout << "    -> Total: " << fixed << setprecision(6) << best_total << " | Avg: " << best_avg
         << ", Iter: " << total_iterations << ", Time: " << setprecision(3) << time_sec << "s\n";

    std::ostringstream els;
    for (size_t i = 0; i < sol.size(); ++i)
    {
        if (i) els << ' ';
        els << sol[i];
    }

    ExperimentResult r;
    r.config = config.name;
    r.file = instance_file;
    r.n = n;
    r.k = k;
    r.alpha = (config.kind == SolverKind::Standard) ? config.alpha : -1.0;
    r.construct_mode = (config.kind == SolverKind::Standard) ? "STANDARD" : "RPG";
    r.ls_mode = "BEST_IMPROVING";
    r.reactive_alphas = "";
    r.reactive_block = "";
    r.sample_size = (config.kind == SolverKind::RPG) ? to_string(config.p) : "";
    r.iterations = MAX_TOTAL_ITERATIONS;
    r.time_limit_s = static_cast<double>(MAX_TIME_MILLIS) / 1000.0;
    r.timed_out = stopped_by_time;
    r.max_value = -best_avg;
    r.size = static_cast<int>(sol.size());
    r.feasible = true;
    r.time_s = time_sec;
    r.elements = els.str();

    return r;
}

bool USE_FILTERS = true;

vector<string> ALLOW_INSTANCES = {"haberman.i"};
vector<int> ALLOW_KS = {3};

// vector<string> ALLOW_CONFIG_PREFIX = {"GRASP_alpha=0.05", "RPG_p=10"};
vector<string> ALLOW_CONFIG_PREFIX = {"GRASP_alpha=0.05"};

vector<string> BLOCK_CONFIG_PREFIX = {};

bool starts_with(const string& s, const string& p) { return s.rfind(p, 0) == 0; }

bool contains_str(const vector<string>& v, const string& x)
{
    for (auto& s : v)
        if (s == x) return true;
    return false;
}

bool contains_int(const vector<int>& v, int x)
{
    for (auto& a : v)
        if (a == x) return true;
    return false;
}

bool match_any_prefix(const string& s, const vector<string>& prefixes)
{
    for (auto& p : prefixes)
        if (starts_with(s, p)) return true;
    return false;
}

bool should_run_config(const string& inst, int k, const ExperimentConfig& cfg)
{
    if (!USE_FILTERS) return true;
    if (!ALLOW_INSTANCES.empty() && !contains_str(ALLOW_INSTANCES, inst)) return false;
    if (!ALLOW_KS.empty() && !contains_int(ALLOW_KS, k)) return false;
    if (!BLOCK_CONFIG_PREFIX.empty() && match_any_prefix(cfg.name, BLOCK_CONFIG_PREFIX))
        return false;
    if (!ALLOW_CONFIG_PREFIX.empty() && !match_any_prefix(cfg.name, ALLOW_CONFIG_PREFIX))
        return false;
    return true;
}

void save_results_to_csv(vector<ExperimentResult>& results, string& output_file)
{
    filesystem::create_directories(filesystem::path(output_file).parent_path());

    ofstream f(output_file, ios::out | ios::trunc);
    if (!f.is_open()) throw runtime_error("Failed to open output CSV: " + output_file);

    f << "config,file,n,k,alpha,construct_mode,ls_mode,reactive_alphas,reactive_block,"
         "sample_size,iterations,time_limit_s,timed_out,max_value,size,feasible,time_s,elements\n";

    f.setf(ios::fixed);
    for (const auto& r : results)
    {
        f << r.config << ',' << r.file << ',' << r.n << ',' << r.k << ',' << setprecision(4)
          << r.alpha << ',' << r.construct_mode << ',' << r.ls_mode << ",\"" << r.reactive_alphas
          << "\"," << r.reactive_block << ',' << r.sample_size << ',' << r.iterations << ','
          << setprecision(0) << r.time_limit_s << ',' << (r.timed_out ? "true" : "false") << ','
          << setprecision(6) << r.max_value << ',' << r.size << ','
          << (r.feasible ? "true" : "false") << ',' << setprecision(3) << r.time_s << ",\""
          << r.elements << "\"\n";
    }
    cout << "\nResults saved to: " << output_file << "\n";
}

void print_summary_statistics(vector<ExperimentResult>& results)
{
    cout << "\n=== SUMMARY STATISTICS ===\n\n";

    vector<pair<string, int>> keys;
    for (auto& r : results) keys.emplace_back(r.file, r.k);
    sort(keys.begin(), keys.end());
    keys.erase(unique(keys.begin(), keys.end()), keys.end());

    for (auto [inst, k] : keys)
    {
        vector<ExperimentResult> subset;
        for (auto& r : results)
            if (r.file == inst && r.k == k) subset.push_back(r);
        
        if (subset.empty()) 
            continue;

        auto best_it = min_element(subset.begin(), subset.end(),
                                   [](const ExperimentResult& a, const ExperimentResult& b)
                                   { return a.max_value > b.max_value; });

        auto& best = *best_it;
        cout << "Instance: " << inst << " | k=" << k << "\n";
        cout << "  Best configuration: " << best.config << "\n";
        cout << "  Best avg cost: " << fixed << setprecision(6) << (-best.max_value) << "\n";
        cout << "  Time: " << setprecision(3) << best.time_s << " seconds\n\n";
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
    string ttt_csv = RESULTS_DIR + "/ttt_grasp_kmedoids_" + string(buf) + ".csv";

    cout << "==============================================\n";
    cout << "GRASP K-MEDOIDS\n";
    cout << "==============================================\n";
    cout << "Tempo máximo por rodada: " << (MAX_TIME_MILLIS / 1000) << " seconds\n";
    cout << "Arquivo de saída: " << output_csv << "\n";
    if (ENABLE_TTT_MODE)
    {
        cout << "TTT MODE: ON\n";
        cout << "  -> runs: " << TTT_RUNS << "\n";
        cout << "  -> targets: por instância × k\n";
        cout << "  -> ttt_csv: " << ttt_csv << "\n";
    }
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

                if (!should_run_config(inst, k, cfg))
                {
                    cout << "  [skip] " << cfg.name << " | k=" << k << " | on " << inst << "\n";
                    continue;
                }

                cout << "\n[Experimento " << exp_count << "/"
                     << (configs.size() * INSTANCE_FILES.size() * K_VALUES.size()) << "]\n";

                auto res = run_experiment(cfg, inst, k, ttt_csv);
                if (!ENABLE_TTT_MODE)
                {
                    all_results.push_back(res);

                    if (exp_count % 10 == 0)
                    {
                        save_results_to_csv(all_results, output_csv);
                    }
                }
            }
        }
    }

    if (!ENABLE_TTT_MODE)
    {
        save_results_to_csv(all_results, output_csv);
    }
    else
    {
        cout << "\nTTT results saved to: " << ttt_csv << "\n";
    }

    auto end = chrono::steady_clock::now();
    double total_time_min = chrono::duration_cast<chrono::seconds>(end - start).count() / 60.0;

    cout << "Experimentos completos\n";
    cout << "Tempo Total: " << fixed << setprecision(2) << total_time_min << " minutos\n";
    if (!ENABLE_TTT_MODE)
    {
        cout << "Resultados salvos: " << output_csv << "\n";
        print_summary_statistics(all_results);
    }
}
