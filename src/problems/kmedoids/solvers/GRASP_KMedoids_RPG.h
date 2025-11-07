#pragma once
#include "problems/kmedoids/solvers/GRASP_KMedoids.h"

class GRASP_KMedoids_RPG : public GRASP_KMedoids
{
   public:
    GRASP_KMedoids_RPG(double alpha, int iterations, const std::vector<std::vector<double>>& D,
                       int k, int p)
        : GRASP_KMedoids(alpha, iterations, D, k), k_local_(k), p_(p)
    {
    }

    Solution<int> constructiveHeuristic() override;

   private:
    int k_local_;  // <- copia do k (para não acessar k_ da base)
    int p_;
};
