#include "GRASP_KMedoids_POP.h"

#include <algorithm>
#include <cmath>

Solution<int> GRASP_KMedoids_POP::constructiveHeuristic()
{
    CL = makeCL();
    RCL = makeRCL();
    sol = createEmptySol();
    cost = numeric_limits<double>::infinity();

    vector<int> triggers;
    triggers.reserve(milestones_.size());
    for (double f : milestones_)
    {
        int t = (int) ceil(f * (double) k_);
        if (t >= 1 && t < k_) 
            triggers.push_back(t);
    }
    sort(triggers.begin(), triggers.end());
    triggers.erase(unique(triggers.begin(), triggers.end()), triggers.end());
    size_t next_tr = 0;

    while ((int) sol->size() < k_ && !CL.empty())
    {
        if (!sol->empty())
        {
            const double c = ObjFunction.evaluate(*sol);
            sol->cost = c;
            cost = c;
        }

        updateCL();
        if (CL.empty())     
            break;

        double min_dc = numeric_limits<double>::infinity();
        double max_dc = -numeric_limits<double>::infinity();
        vector<double> deltas(CL.size(), 0.0);

        for (size_t i = 0; i < CL.size(); ++i)
        {
            int c = CL[i];
            double dc = ObjFunction.evaluate_insertion_cost(c, *sol);
            deltas[i] = dc;
            if (dc < min_dc) 
                min_dc = dc;
            if (dc > max_dc) 
                max_dc = dc;
        }

        RCL.clear();
        double thresh = (max_dc > min_dc) ? (min_dc + alpha * (max_dc - min_dc)) : min_dc;

        for (size_t i = 0; i < CL.size(); ++i)
        {
            if (deltas[i] <= thresh) 
                RCL.push_back(CL[i]);
        }

        int chosen;
        if (RCL.empty())
        {
            size_t best_idx = 0;
            for (size_t i = 1; i < CL.size(); ++i)
                if (deltas[i] < deltas[best_idx]) 
                    best_idx = i;
            chosen = CL[best_idx];
            CL.erase(CL.begin() + (long) best_idx);
        }
        else
        {
            uniform_int_distribution<size_t> dist(0, RCL.size() - 1);
            chosen = RCL[dist(rng_)];
            auto it = find(CL.begin(), CL.end(), chosen);
            if (it != CL.end()) 
                CL.erase(it);
        }

        sol->add(chosen);
        sol->cost = ObjFunction.evaluate(*sol);
        RCL.clear();

        if (next_tr < triggers.size() && (int) sol->size() == triggers[next_tr])
        {
            localSearch();
            ++next_tr;
        }
    }

    return *sol;
}
