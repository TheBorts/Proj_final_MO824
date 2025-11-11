#pragma once
#include "problems/kmedoids/solvers/GRASP_KMedoids.h"

using namespace std;

class GRASP_KMedoids_WLS : public GRASP_KMedoids
{
   public:
    enum class LSSearch
    {
        BestImproving,
        FirstImproving
    };

    GRASP_KMedoids_WLS(double alpha, int iterations, const vector<vector<double>>& D, int k,
                      LSSearch mode = LSSearch::BestImproving);

    Solution<int> localSearch() override;

   private:
    const vector<vector<double>>& D_;
    int n_, m_, k_;
    LSSearch mode_;


    vector<int> assignments;
    vector<double> summed_distances;

    void buildAssignments(const vector<int>& S);
    void buildSummedDistances(const vector<int>& S);


    void iterateConvergence(vector<int>& S);
    void updateSummedDistances_swappoint(int p_out, int where_to, const vector<int>& S);
};
