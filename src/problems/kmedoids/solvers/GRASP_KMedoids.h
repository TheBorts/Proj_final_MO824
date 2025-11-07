#pragma once
#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "../../../metaheuristics/grasp/AbstractGRASP.h"
#include "../../../solutions/Solution.h"
#include "../KMedoidsEvaluator.h"

using namespace std;

class GRASP_KMedoids : public AbstractGRASP<int>
{
   public:
    GRASP_KMedoids(double alpha, int iterations, const vector<vector<double>>& D, int k);

    vector<int> makeCL() override;
    vector<int> makeRCL() override;
    void updateCL() override;
    Solution<int> createEmptySol() override;
    Solution<int> localSearch() override;
    Solution<int> constructiveHeuristic() override;

   private:
    const vector<vector<double>> D_;
    const int n_;
    const int k_;

    KMedoidsEvaluator evaluator_;

    mt19937& rng_ = AbstractGRASP<int>::rng;
};
