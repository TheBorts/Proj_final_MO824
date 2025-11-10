#include "GRASP_KMedoids_RPG.h"

#include <algorithm>
#include <random>

Solution<int> GRASP_KMedoids_RPG::constructiveHeuristic()
{
    CL = makeCL();
    RCL = makeRCL();
    sol = createEmptySol();
    cost = numeric_limits<double>::infinity();

    auto& rng = AbstractGRASP<int>::rng;

    while (static_cast<int>(sol->size()) < k_local_ && !CL.empty())
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

        const int m = min<int>(p_, static_cast<int>(CL.size()));
        shuffle(CL.begin(), CL.end(), rng);
        vector<int> sample(CL.begin(), CL.begin() + m);

        int chosen = sample[0];
        double best_dc = ObjFunction.evaluate_insertion_cost(chosen, *sol);
        for (int i = 1; i < m; ++i)
        {
            int cnd = sample[i];
            double dc = ObjFunction.evaluate_insertion_cost(cnd, *sol);
            if (dc < best_dc)
            {
                best_dc = dc;
                chosen = cnd;
            }
        }

        auto it = find(CL.begin(), CL.end(), chosen);
        if (it != CL.end()) 
            CL.erase(it);

        sol->add(chosen);
        sol->cost = ObjFunction.evaluate(*sol);
        RCL.clear();
    }
    return *sol;
}
