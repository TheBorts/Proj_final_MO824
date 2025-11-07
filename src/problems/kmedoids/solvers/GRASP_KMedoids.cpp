#include "GRASP_KMedoids.h"

#include <iostream>
#include <numeric>
#include <unordered_set>

GRASP_KMedoids::GRASP_KMedoids(double alpha, int iterations, const vector<vector<double>>& D, int k)
    : AbstractGRASP<int>(evaluator_, alpha, iterations),
      D_(D),
      n_(static_cast<int>(D.size())),
      k_(k),
      evaluator_(D, k)
{
}

vector<int> GRASP_KMedoids::makeCL()
{
    vector<int> cl(n_);
    iota(cl.begin(), cl.end(), 0);
    return cl;
}

vector<int> GRASP_KMedoids::makeRCL() { return {}; }

void GRASP_KMedoids::updateCL()
{
    if (!sol.has_value()) return;
    if (CL.empty()) return;

    unordered_set<int> in_sol(sol->begin(), sol->end());
    vector<int> filtered;
    filtered.reserve(CL.size());
    for (int c : CL)
    {
        if (in_sol.find(c) == in_sol.end()) filtered.push_back(c);
    }
    CL.swap(filtered);
}

Solution<int> GRASP_KMedoids::createEmptySol()
{
    Solution<int> s;
    s.cost = numeric_limits<double>::infinity();
    return s;
}

Solution<int> GRASP_KMedoids::constructiveHeuristic()
{
    CL = makeCL();
    RCL = makeRCL();
    sol = createEmptySol();
    cost = numeric_limits<double>::infinity();

    while (static_cast<int>(sol->size()) < k_ && !CL.empty())
    {
        if (!sol->empty())
        {
            const double c = ObjFunction.evaluate(*sol);
            sol->cost = c;
            cost = c;
        }

        updateCL();
        if (CL.empty()) break;

        double min_dc = numeric_limits<double>::infinity();
        double max_dc = -numeric_limits<double>::infinity();
        vector<double> deltas(CL.size(), 0.0);

        for (size_t i = 0; i < CL.size(); ++i)
        {
            int c = CL[i];
            double dc = ObjFunction.evaluate_insertion_cost(c, *sol);
            deltas[i] = dc;
            if (dc < min_dc) min_dc = dc;
            if (dc > max_dc) max_dc = dc;
        }

        RCL.clear();
        double thresh = (max_dc > min_dc) ? (min_dc + alpha * (max_dc - min_dc)) : min_dc;

        for (size_t i = 0; i < CL.size(); ++i)
        {
            if (deltas[i] <= thresh) RCL.push_back(CL[i]);
        }

        int chosen;
        if (RCL.empty())
        {
            size_t best_idx = 0;
            for (size_t i = 1; i < CL.size(); ++i)
            {
                if (deltas[i] < deltas[best_idx]) best_idx = i;
            }
            chosen = CL[best_idx];
            CL.erase(CL.begin() + static_cast<long>(best_idx));
        }
        else
        {
            uniform_int_distribution<size_t> dist(0, RCL.size() - 1);
            chosen = RCL[dist(rng_)];
            auto it = find(CL.begin(), CL.end(), chosen);
            if (it != CL.end()) CL.erase(it);
        }

        sol->add(chosen);
        double c = ObjFunction.evaluate(*sol);
        sol->cost = c;
        RCL.clear();
    }

    return *sol;
}

Solution<int> GRASP_KMedoids::localSearch()
{
    double eps = 1e-12;
    bool improved = true;

    while (improved)
    {
        improved = false;
        double best_dc = 0.0;
        int best_in = -1, best_out = -1;

        updateCL();
        vector<int> out_list(sol->begin(), sol->end());
        vector<int> in_list = CL;

        for (int cin : in_list)
        {
            for (int cout : out_list)
            {
                double dc = ObjFunction.evaluate_exchange_cost(cin, cout, *sol);
                if (dc < best_dc - eps)
                {
                    best_dc = dc;
                    best_in = cin;
                    best_out = cout;
                }
            }
        }

        if (best_in != -1 && best_out != -1)
        {
            auto oit = find(sol->begin(), sol->end(), best_out);
            if (oit != sol->end()) sol->erase(oit);
            sol->add(best_in);

            CL.push_back(best_out);
            auto cit = find(CL.begin(), CL.end(), best_in);
            if (cit != CL.end()) CL.erase(cit);

            double c = ObjFunction.evaluate(*sol);
            sol->cost = c;
            improved = true;
        }
    }

    return *sol;
}
