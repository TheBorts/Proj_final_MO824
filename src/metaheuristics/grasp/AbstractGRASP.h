// AbstractGRASP.h
#pragma once
#include <vector>
#include <random>
#include <limits>
#include <optional>
#include <algorithm>
#include <iostream>

#include "../../problems/Evaluator.h"
#include "../../solutions/Solution.h"

template <typename E>
class AbstractGRASP {
    public:
        static inline bool verbose = true;
        static inline std::mt19937 rng{0};

        Evaluator<E>& ObjFunction;
        double alpha;
        std::optional<double> bestCost;
        std::optional<double> cost;
        std::optional<Solution<E>> bestSol;
        std::optional<Solution<E>> sol;
        int iterations;
        std::vector<E> CL;
        std::vector<E> RCL;

        AbstractGRASP(Evaluator<E>& obj_function, double alpha_, int iterations_)
            : ObjFunction(obj_function),
            alpha(alpha_),
            bestCost(std::nullopt),
            cost(std::nullopt),
            bestSol(std::nullopt),
            sol(std::nullopt),
            iterations(iterations_) {}

        virtual ~AbstractGRASP() = default;

        virtual std::vector<E> makeCL() = 0;

        virtual std::vector<E> makeRCL() = 0;

        virtual void updateCL() = 0;

        virtual Solution<E> createEmptySol() = 0;

        virtual Solution<E> localSearch() = 0;

        virtual Solution<E> constructiveHeuristic() {
            CL  = makeCL();
            RCL = makeRCL();
            sol = createEmptySol();
            cost = std::numeric_limits<double>::infinity();

            while (!constructiveStopCriteria()) {
                if (CL.empty()) break;

                double max_cost = -std::numeric_limits<double>::infinity();
                double min_cost =  std::numeric_limits<double>::infinity();

                cost = ObjFunction.evaluate(*sol);

                updateCL();
                if (CL.empty()) 
                    break;

                for (auto& c : CL) {
                    double delta = ObjFunction.evaluate_insertion_cost(c, *sol);
                    if (delta < min_cost) min_cost = delta;
                    if (delta > max_cost) max_cost = delta;
                }

                RCL.clear();
                double threshold = min_cost + alpha * (max_cost - min_cost);
                for (auto& c : CL) {
                    double delta = ObjFunction.evaluate_insertion_cost(c, *sol);
                    if (delta <= threshold) RCL.push_back(c);
                }
                if (RCL.empty()) break;

                std::uniform_int_distribution<std::size_t> dist(0, RCL.size() - 1);
                E in_cand = RCL[dist(rng)];

                auto it = std::find(CL.begin(), CL.end(), in_cand);
                if (it != CL.end()) CL.erase(it);
                sol->add(in_cand);
                ObjFunction.evaluate(*sol);
                RCL.clear();
            }

            return *sol;
        }

        Solution<E> solve() {
            bestSol = createEmptySol();
            for (int i = 0; i < iterations; ++i) {
                constructiveHeuristic();
                localSearch();

                if (bestSol->cost > sol->cost) {
                    bestSol = Solution<E>(*sol);
                    if (verbose) {
                        std::cout << "(Iter. " << i << ") BestSol = " << *bestSol << "\n";
                    }
                }
            }
            return *bestSol;
        }

        bool constructiveStopCriteria() const {
            if (!cost.has_value() || !sol.has_value()) 
                return false;
            return !(cost.value() > sol->cost);
        }

        static void set_seed(unsigned seed) { rng.seed(seed); }
};
