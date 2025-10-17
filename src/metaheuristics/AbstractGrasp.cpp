#ifndef ABSTRACT_GRASP_H
#define ABSTRACT_GRASP_H

#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include <iostream>
#include <memory>

template <typename E> class Evaluator;

template <typename E> class Solution {
public:
    std::vector<E> elements;
    double cost = std::numeric_limits<double>::infinity();

    Solution() = default;

    Solution(const Solution<E>& other) {
        elements = other.elements;
        cost = other.cost;
    }

    void add(const E& e) {
        elements.push_back(e);
    }
};

template <typename E> class BiasFunction {
public:
    virtual int selectCandidate(const std::vector<E>& RCL, Evaluator<E>* objFunction, Solution<E>* sol) = 0;
};

template <typename E> class Evaluator {
public:
    virtual double evaluate(Solution<E>* sol) = 0;
    virtual double evaluateInsertionCost(const E& e, Solution<E>* sol) = 0;
};

template <typename E> class AbstractGRASP {
public:
    static bool verbose;
    static std::mt19937 rng;

    Evaluator<E>* ObjFunction;
    double alpha;
    double bestCost;
    double cost;
    Solution<E> bestSol;
    Solution<E> sol;
    int seconds;

    std::vector<E> CL;
    std::vector<E> RCL;

    BiasFunction<E>* biasFunction;
    bool isFirstImproving;

    AbstractGRASP(Evaluator<E>* objFunction, double alpha, int seconds, BiasFunction<E>* biasFunction, bool isFirstImproving)
        : ObjFunction(objFunction), alpha(alpha), seconds(seconds), biasFunction(biasFunction), isFirstImproving(isFirstImproving)
    {}

    virtual std::vector<E> makeCL() = 0;
    virtual std::vector<E> makeRCL() = 0;
    virtual void updateCL() = 0;
    virtual Solution<E> createEmptySol() = 0;
    virtual Solution<E> localSearch() = 0;

    Solution<E> constructiveHeuristic(int numberOfRandomIterations) {
        CL = makeCL();
        RCL = makeRCL();
        sol = createEmptySol();
        cost = std::numeric_limits<double>::infinity();
        int currentIteration = 0;

        while (!constructiveStopCriteria()) {
            double maxCost = -std::numeric_limits<double>::infinity();
            double minCost = std::numeric_limits<double>::infinity();
            cost = ObjFunction->evaluate(&sol);
            updateCL();

            if (CL.empty()) break;

            E bestCandidate = CL[0];

            for (const E& c : CL) {
                double deltaCost = ObjFunction->evaluateInsertionCost(c, &sol);
                if (deltaCost < minCost) {
                    minCost = deltaCost;
                    bestCandidate = c;
                }
                if (deltaCost > maxCost) {
                    maxCost = deltaCost;
                }
            }

            if (!std::isfinite(minCost) || !std::isfinite(maxCost)) break;

            RCL.clear();
            for (const E& c : CL) {
                double deltaCost = ObjFunction->evaluateInsertionCost(c, &sol);
                if (deltaCost <= minCost + alpha * (maxCost - minCost)) {
                    RCL.push_back(c);
                }
            }

            std::sort(RCL.begin(), RCL.end(), [&](const E& a, const E& b) {
                return ObjFunction->evaluateInsertionCost(a, &sol) < ObjFunction->evaluateInsertionCost(b, &sol);
            });

            E inCand = bestCandidate;

            if (RCL.empty()) break;

            if (currentIteration < numberOfRandomIterations || numberOfRandomIterations == -1) {
                int indexCandidateToEnterSolution = biasFunction->selectCandidate(RCL, ObjFunction, &sol);
                inCand = RCL[indexCandidateToEnterSolution];
            }

            CL.erase(std::remove(CL.begin(), CL.end(), inCand), CL.end());
            sol.add(inCand);

            ObjFunction->evaluate(&sol);
            RCL.clear();

            currentIteration++;
        }

        return sol;
    }

    Solution<E> solve(int numberOfRandomIterations) {
        bestSol = createEmptySol();
        bestSol.cost = std::numeric_limits<double>::infinity();

        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            auto endTime = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
            if (elapsed >= seconds) break;

            constructiveHeuristic(numberOfRandomIterations);
            localSearch();

            if (bestSol.cost > sol.cost) {
                bestSol = sol;
                if (verbose) {
                    std::cout << "BestSol cost = " << bestSol.cost << std::endl;
                }
            }
        }

        return bestSol;
    }

    bool constructiveStopCriteria() {
        return cost < sol.cost;
    }
};

template <typename E>
bool AbstractGRASP<E>::verbose = true;

template <typename E>
std::mt19937 AbstractGRASP<E>::rng(0);

#endif
