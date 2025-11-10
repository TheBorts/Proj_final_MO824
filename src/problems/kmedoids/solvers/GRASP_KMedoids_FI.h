#pragma once
#include "problems/kmedoids/solvers/GRASP_KMedoids.h"

using namespace std;

class GRASP_KMedoids_FI : public GRASP_KMedoids
{
   public:
    GRASP_KMedoids_FI(double alpha, int iterations, const std::vector<std::vector<double>>& D,
                      int k)
        : GRASP_KMedoids(alpha, iterations, D, k)
    {
    }

    Solution<int> localSearch() override;
};
