#pragma once
#include "problems/kmedoids/solvers/GRASP_KMedoids.h"

using namespace std;

class GRASP_KMedoids_RW : public GRASP_KMedoids
{
   public:
    enum class LSSearch
    {
        BestImproving,
        FirstImproving
    };

    GRASP_KMedoids_RW(double alpha, int iterations, const vector<vector<double>>& D, int k,
                      LSSearch mode = LSSearch::BestImproving);

    Solution<int> localSearch() override;

   private:
    const vector<vector<double>>& D_;
    int n_, m_, k_;
    LSSearch mode_;

    vector<int> phi1;
    vector<double> d1, d2;
    vector<vector<int>> servedBy;
    vector<double> gain;
    vector<double> loss;

    void buildAssignments(const vector<int>& S);
    void buildGainLoss(const vector<int>& S);
    double extra_pair(int fi, int fr, const vector<int>& S);
    void apply_swap_update(int fi, int fr, vector<int>& S);
    bool in_solution(int f, const vector<int>& S);
};
