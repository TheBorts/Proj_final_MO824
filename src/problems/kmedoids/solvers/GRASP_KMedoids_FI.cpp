#include "GRASP_KMedoids_FI.h"

#include <algorithm>

Solution<int> GRASP_KMedoids_FI::localSearch()
{
    const double eps = 1e-12;
    bool improved = true;

    while (improved)
    {
        improved = false;

        updateCL();
        vector<int> out_list(sol->begin(), sol->end());
        vector<int> in_list = CL;

        bool found = false;
        int best_in = -1, best_out = -1;

        for (int cin : in_list)
        {
            for (int cout : out_list)
            {
                double dc = ObjFunction.evaluate_exchange_cost(cin, cout, *sol);
                if (dc < -eps)
                {
                    best_in = cin;
                    best_out = cout;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }

        if (found)
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
