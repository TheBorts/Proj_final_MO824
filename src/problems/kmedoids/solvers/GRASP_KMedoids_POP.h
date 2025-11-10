#pragma once
#include "problems/kmedoids/solvers/GRASP_KMedoids.h"

using namespace std;

class GRASP_KMedoids_POP : public GRASP_KMedoids
{
   public:
    GRASP_KMedoids_POP(double alpha, int iterations, const std::vector<std::vector<double>>& D,
                       int k, std::vector<double> milestones = {0.40, 0.80})
        : GRASP_KMedoids(alpha, iterations, D, k), milestones_(std::move(milestones))
    {
    }

    Solution<int> constructiveHeuristic() override;

   private:
    std::vector<double> milestones_;
};
