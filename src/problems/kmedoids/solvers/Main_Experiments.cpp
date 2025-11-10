#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "metaheuristics/grasp/AbstractGRASP.h"
#include "problems/kmedoids/common.h"
#include "problems/kmedoids/solvers/GRASP_KMedoids.h"
#include "problems/kmedoids/solvers/GRASP_KMedoids_FI.h"
#include "problems/kmedoids/solvers/GRASP_KMedoids_POP.h"
#include "problems/kmedoids/solvers/GRASP_KMedoids_RPG.h"
#include "problems/kmedoids/solvers/GRASP_KMedoids_RW.h"

using namespace std;

string INSTANCES_DIR = "instances/general";

vector<string> INSTANCE_FILES = {"2-FACE.i",
                                 "200DATA.I",
                                 "400p3c.i",
                                 "A1.I",
                                 "BreastB.I",
                                 "Compound.I",
                                 "Concrete_Data.i",
                                 "DBLCA.I",
                                 "DBLCB.I",
                                 "DOWJONES.I",
                                 "Normal300.i",
                                 "PARKINSONS.i",
                                 "Prima_Indians.i",
                                 "SPRDESP.I",
                                 "TRIPADVISOR.I",
                                 "Uniform400.i",
                                 "Uniform700.i",
                                 "aggregation.I",
                                 "banknote_authentication.I",
                                 "broken-ring.i",
                                 "bupa.I",
                                 "chart.i",
                                 "ecoli.i",
                                 "face.i",
                                 "forestfires.I",
                                 "gauss9.i",
                                 "glass.I",
                                 "haberman.i",
                                 "hayes-roth.I",
                                 "indian.i",
                                 "indochina_combat.I",
                                 "ionosphere.I",
                                 "iris.i",
                                 "maronna.i",
                                 "moreshapes.i",
                                 "new-thyroid.I",
                                 "numbers2.i",
                                 "outliers.i",
                                 "pib100.i",
                                 "ruspini.i",
                                 "sonar.I",
                                 "spherical_4d3c.i",
                                 "synthetic_control.i",
                                 "vowel2.i",
                                 "wdbc.I",
                                 "wine.i",
                                 "yeast.i",
                                 "waveform.I"};


bool USE_ZSCORE = true;

vector<int> K_VALUES = {3, 4, 5, 6};

vector<double> ALPHA_VALUES = {0.05};
vector<int> P_VALUES = {5, 10, 20};

long MAX_TIME_MILLIS = 15 * 60L * 1000L;
int MAX_TOTAL_ITERATIONS = 10000;

string RESULTS_DIR = "results";

bool ENABLE_TTT_MODE = false;
int TTT_RUNS = 50;

vector<tuple<string, int, double>> TTT_TARGETS = {
    // 2-FACE.i
    make_tuple("2-FACE.i", 3, 0.6777849370076154),
    make_tuple("2-FACE.i", 4, 0.5925680856930263),
    make_tuple("2-FACE.i", 5, 0.5136686686860198),
    make_tuple("2-FACE.i", 6, 0.4551875829945722),

    // 200DATA.I
    make_tuple("200DATA.I", 3, 0.3717549369520792),
    make_tuple("200DATA.I", 4, 0.26560250614722736),
    make_tuple("200DATA.I", 5, 0.23349404604831522),
    make_tuple("200DATA.I", 6, 0.2130904365380815),

    // 400p3c.i
    make_tuple("400p3c.i", 3, 0.2803071863095197),
    make_tuple("400p3c.i", 4, 0.23913301191729872),
    make_tuple("400p3c.i", 5, 0.21291918772913185),
    make_tuple("400p3c.i", 6, 0.19321313691359626),

    // A1.I (FAILED_PROCESS)
    make_tuple("A1.I", 3, 2.50),
    make_tuple("A1.I", 4, 2.50),
    make_tuple("A1.I", 5, 2.50),
    make_tuple("A1.I", 6, 2.50),

    // BreastB.I
    make_tuple("BreastB.I", 3, 38.30834666746651),
    make_tuple("BreastB.I", 4, 36.947697490825476),
    make_tuple("BreastB.I", 5, 35.72473085615393),
    make_tuple("BreastB.I", 6, 34.53491837195962),

    // Compound.I
    make_tuple("Compound.I", 3, 0.5607496041714434),
    make_tuple("Compound.I", 4, 0.4944654005331013),
    make_tuple("Compound.I", 5, 0.4419115090351092),
    make_tuple("Compound.I", 6, 0.4003393271379871),

    // Concrete_Data.i
    make_tuple("Concrete_Data.i", 3, 2.472333919952707),
    make_tuple("Concrete_Data.i", 4, 2.2994475931633764),
    make_tuple("Concrete_Data.i", 5, 2.175513956264638),
    make_tuple("Concrete_Data.i", 6, 2.0611661040879494),

    // DBLCA.I
    make_tuple("DBLCA.I", 3, 27.24731249102294),
    make_tuple("DBLCA.I", 4, 26.767559862094686),
    make_tuple("DBLCA.I", 5, 26.31088195716591),
    make_tuple("DBLCA.I", 6, 25.89554395843004),

    // DBLCB.I
    make_tuple("DBLCB.I", 3, 27.18407274000569),
    make_tuple("DBLCB.I", 4, 26.390603404757606),
    make_tuple("DBLCB.I", 5, 25.93097987256351),
    make_tuple("DBLCB.I", 6, 25.497571028991725),

    // DOWJONES.I
    make_tuple("DOWJONES.I", 3, 0.62839770556556),
    make_tuple("DOWJONES.I", 4, 0.46450525836012274),
    make_tuple("DOWJONES.I", 5, 0.32846508245921474),
    make_tuple("DOWJONES.I", 6, 0.2697104243904902),

    // Normal300.i
    make_tuple("Normal300.i", 3, 0.8192645852107954),
    make_tuple("Normal300.i", 4, 0.7163812113566994),
    make_tuple("Normal300.i", 5, 0.6566305868306369),
    make_tuple("Normal300.i", 6, 0.6047651113513913),

    // PARKINSONS.i
    make_tuple("PARKINSONS.i", 3, 3.4218968153232394),
    make_tuple("PARKINSONS.i", 4, 3.156824670932389),
    make_tuple("PARKINSONS.i", 5, 3.0179290158673573),
    make_tuple("PARKINSONS.i", 6, 2.8824580192663385),

    // Prima_Indians.i
    make_tuple("Prima_Indians.i", 3, 2.210191605586158),
    make_tuple("Prima_Indians.i", 4, 2.089748854418461),
    make_tuple("Prima_Indians.i", 5, 1.993312307956293),
    make_tuple("Prima_Indians.i", 6, 1.9004509011146042),

    // SPRDESP.I
    make_tuple("SPRDESP.I", 3, 0.6849917614028074),
    make_tuple("SPRDESP.I", 4, 0.6193669823411156),
    make_tuple("SPRDESP.I", 5, 0.5635163785051309),
    make_tuple("SPRDESP.I", 6, 0.5117847195917843),

    // TRIPADVISOR.I
    make_tuple("TRIPADVISOR.I", 3, 2.6138658133606154),
    make_tuple("TRIPADVISOR.I", 4, 2.4949413795520257),
    make_tuple("TRIPADVISOR.I", 5, 2.414470975993326),
    make_tuple("TRIPADVISOR.I", 6, 2.3500075115552184),

    // Uniform400.i
    make_tuple("Uniform400.i", 3, 0.7961680236260879),
    make_tuple("Uniform400.i", 4, 0.6559576382626272),
    make_tuple("Uniform400.i", 5, 0.5971898712162732),
    make_tuple("Uniform400.i", 6, 0.5434664362821542),

    // Uniform700.i
    make_tuple("Uniform700.i", 3, 0.7829947905524188),
    make_tuple("Uniform700.i", 4, 0.6578541218552572),
    make_tuple("Uniform700.i", 5, 0.5990595740733328),
    make_tuple("Uniform700.i", 6, 0.5463500529590206),

    // aggregation.I
    make_tuple("aggregation.I", 3, 0.6738004841194628),
    make_tuple("aggregation.I", 4, 0.5353234030082256),
    make_tuple("aggregation.I", 5, 0.46925832922536576),
    make_tuple("aggregation.I", 6, 0.42397399234325933),

    // banknote_authentication.I
    make_tuple("banknote_authentication.I", 3, 1.2417515178562275),
    make_tuple("banknote_authentication.I", 4, 1.1118423919236209),
    make_tuple("banknote_authentication.I", 5, 1.01025739903814),
    make_tuple("banknote_authentication.I", 6, 0.9380113604319321),

    // broken-ring.i
    make_tuple("broken-ring.i", 3, 0.8038581993201405),
    make_tuple("broken-ring.i", 4, 0.6089207047151881),
    make_tuple("broken-ring.i", 5, 0.5150057404624077),
    make_tuple("broken-ring.i", 6, 0.4683257723724673),

    // bupa.I
    make_tuple("bupa.I", 3, 1.761511782460369),
    make_tuple("bupa.I", 4, 1.6599017348379654),
    make_tuple("bupa.I", 5, 1.571809234694991),
    make_tuple("bupa.I", 6, 1.5148520234578389),

    // chart.i
    make_tuple("chart.i", 3, 7.708550436452115),
    make_tuple("chart.i", 4, 7.598374794515124),
    make_tuple("chart.i", 5, 7.522369733946849),
    make_tuple("chart.i", 6, 7.451489621041898),

    // ecoli.i
    make_tuple("ecoli.i", 3, 1.5610428658490456),
    make_tuple("ecoli.i", 4, 1.43857707996486),
    make_tuple("ecoli.i", 5, 1.360998544546864),
    make_tuple("ecoli.i", 6, 1.2889553880093576),

    // face.i
    make_tuple("face.i", 3, 0.6675113981051324),
    make_tuple("face.i", 4, 0.5829653915322806),
    make_tuple("face.i", 5, 0.521134226286712),
    make_tuple("face.i", 6, 0.46578916945272403),

    // forestfires.I
    make_tuple("forestfires.I", 3, 1.8502355599871445),
    make_tuple("forestfires.I", 4, 1.731387006242273),
    make_tuple("forestfires.I", 5, 1.6584857776716169),
    make_tuple("forestfires.I", 6, 1.590278197896334),

    // gauss9.i
    make_tuple("gauss9.i", 3, 0.8125990864981274),
    make_tuple("gauss9.i", 4, 0.6739554072349321),
    make_tuple("gauss9.i", 5, 0.5886885483326536),
    make_tuple("gauss9.i", 6, 0.5207377704445877),

    // glass.I
    make_tuple("glass.I", 3, 1.9433828548100605),
    make_tuple("glass.I", 4, 1.7564048583724667),
    make_tuple("glass.I", 5, 1.6468437700196688),
    make_tuple("glass.I", 6, 1.5533752066706112),

    // haberman.i
    make_tuple("haberman.i", 3, 1.1274192065879267),
    make_tuple("haberman.i", 4, 0.9849232343256404),
    make_tuple("haberman.i", 5, 0.8902673949658259),
    make_tuple("haberman.i", 6, 0.8310722392975113),

    // hayes-roth.I
    make_tuple("hayes-roth.I", 3, 2.1019254032385977),
    make_tuple("hayes-roth.I", 4, 1.9792006943015499),
    make_tuple("hayes-roth.I", 5, 1.8599036985160737),
    make_tuple("hayes-roth.I", 6, 1.7749837013037997),

    // indian.i
    make_tuple("indian.i", 3, 2.005019121671649),
    make_tuple("indian.i", 4, 1.8804765664692185),
    make_tuple("indian.i", 5, 1.7988273763036768),
    make_tuple("indian.i", 6, 1.725689226830192),

    // indochina_combat.I
    make_tuple("indochina_combat.I", 3, 1.116419365928941),
    make_tuple("indochina_combat.I", 4, 1.0056941181812202),
    make_tuple("indochina_combat.I", 5, 0.9126667414168596),
    make_tuple("indochina_combat.I", 6, 0.8255441652699176),

    // ionosphere.I
    make_tuple("ionosphere.I", 3, 4.154197151125744),
    make_tuple("ionosphere.I", 4, 3.9026068478334617),
    make_tuple("ionosphere.I", 5, 3.7515144356685406),
    make_tuple("ionosphere.I", 6, 3.6089705558703957),

    // iris.i
    make_tuple("iris.i", 3, 0.8722134642882984),
    make_tuple("iris.i", 4, 0.7797447613078735),
    make_tuple("iris.i", 5, 0.6979250860391732),
    make_tuple("iris.i", 6, 0.6539037491297215),

    // maronna.i
    make_tuple("maronna.i", 3, 0.6956626104542487),
    make_tuple("maronna.i", 4, 0.4728087794006858),
    make_tuple("maronna.i", 5, 0.4296678720556892),
    make_tuple("maronna.i", 6, 0.39007068275627743),

    // moreshapes.i
    make_tuple("moreshapes.i", 3, 0.6347621636283203),
    make_tuple("moreshapes.i", 4, 0.40742502638788025),
    make_tuple("moreshapes.i", 5, 0.2935808705473496),
    make_tuple("moreshapes.i", 6, 0.23808790348868814),

    // new-thyroid.I
    make_tuple("new-thyroid.I", 3, 1.2144585702871928),
    make_tuple("new-thyroid.I", 4, 1.0768561803104653),
    make_tuple("new-thyroid.I", 5, 0.9841436126605825),
    make_tuple("new-thyroid.I", 6, 0.9254939857085749),

    // numbers2.i
    make_tuple("numbers2.i", 3, 0.7797840602070729),
    make_tuple("numbers2.i", 4, 0.6016094536844063),
    make_tuple("numbers2.i", 5, 0.5253186054496919),
    make_tuple("numbers2.i", 6, 0.44840731047541127),

    // outliers.i
    make_tuple("outliers.i", 3, 0.7322827353829233),
    make_tuple("outliers.i", 4, 0.636728089268345),
    make_tuple("outliers.i", 5, 0.5760946274681759),
    make_tuple("outliers.i", 6, 0.5184221556330948),

    // pib100.i
    make_tuple("pib100.i", 3, 0.6304994587963254),
    make_tuple("pib100.i", 4, 0.5517848038502665),
    make_tuple("pib100.i", 5, 0.4929078412014404),
    make_tuple("pib100.i", 6, 0.43405554820629644),

    // ruspini.i
    make_tuple("ruspini.i", 3, 0.5983211624530832),
    make_tuple("ruspini.i", 4, 0.31870564898289117),
    make_tuple("ruspini.i", 5, 0.2874573228645719),
    make_tuple("ruspini.i", 6, 0.2565927754786713),

    // sonar.I
    make_tuple("sonar.I", 3, 7.316592005308365),
    make_tuple("sonar.I", 4, 7.107490288554134),
    make_tuple("sonar.I", 5, 6.918121506010238),
    make_tuple("sonar.I", 6, 6.726095930009992),

    // spherical_4d3c.i
    make_tuple("spherical_4d3c.i", 3, 0.5604520670015242),
    make_tuple("spherical_4d3c.i", 4, 0.333503091990253),
    make_tuple("spherical_4d3c.i", 5, 0.3158341693872563),
    make_tuple("spherical_4d3c.i", 6, 0.29977695035703683),

    // synthetic_control.i
    make_tuple("synthetic_control.i", 3, 7.105057383597926),
    make_tuple("synthetic_control.i", 4, 7.005452811893172),
    make_tuple("synthetic_control.i", 5, 6.925284603444955),
    make_tuple("synthetic_control.i", 6, 6.860732556947375),

    // vowel2.i
    make_tuple("vowel2.i", 3, 0.8034479659359794),
    make_tuple("vowel2.i", 4, 0.6961640549780462),
    make_tuple("vowel2.i", 5, 0.6036748387042824),
    make_tuple("vowel2.i", 6, 0.5334833172276681),

    // wdbc.I
    make_tuple("wdbc.I", 3, 3.9912681484152666),
    make_tuple("wdbc.I", 4, 3.8404766432702044),
    make_tuple("wdbc.I", 5, 3.7148914402257565),
    make_tuple("wdbc.I", 6, 3.6003277314901982),

    // wine.i
    make_tuple("wine.i", 3, 2.806292747636261),
    make_tuple("wine.i", 4, 2.6745322137761507),
    make_tuple("wine.i", 5, 2.5713838688951274),
    make_tuple("wine.i", 6, 2.4883597120523286),

    // yeast.i
    make_tuple("yeast.i", 3, 1.8295161137229363),
    make_tuple("yeast.i", 4, 1.7324450579755923),
    make_tuple("yeast.i", 5, 1.643711529335192),
    make_tuple("yeast.i", 6, 1.5676884753450688),

    // waveform.I (FAILED_PROCESS)
    make_tuple("waveform.I", 3, 2.50),
    make_tuple("waveform.I", 4, 2.50),
    make_tuple("waveform.I", 5, 2.50),
    make_tuple("waveform.I", 6, 2.50),
};

double get_target_avg_for(string& file, int k)
{
    for (auto& t : TTT_TARGETS)
        if (get<0>(t) == file && get<1>(t) == k) return get<2>(t);
    return -1.0;
}

void silence_framework() { AbstractGRASP<int>::verbose = false; }

enum class SolverKind
{
    Standard,
    RPG,
    StandardFI,
    POP,
    RW_BI
};

struct ExperimentConfig
{
    string name;
    SolverKind kind;
    double alpha;
    int p;
};

struct ExperimentResult
{
    string config;
    string file;
    int n{};
    int k{};
    double alpha{};
    string construct_mode;
    string ls_mode;
    string reactive_alphas;
    string reactive_block;
    string sample_size;
    int iterations{};
    double time_limit_s{};
    bool timed_out{};
    double max_value{};
    int size{};
    bool feasible{};
    double time_s{};
    string elements;
};

class GRASP_KMedoids_WithStopping : public GRASP_KMedoids
{
   public:
    GRASP_KMedoids_WithStopping(double alpha, int iterations, vector<vector<double>>& D, int k,
                                long max_time_ms, double target_avg_value = -1.0,
                                bool ttt_mode = false)
        : GRASP_KMedoids(alpha, iterations, D, k),
          max_time_ms_(max_time_ms),
          target_avg_value_(target_avg_value),
          ttt_mode_(ttt_mode)
    {
    }

    Solution<int> solve()
    {
        bestSol = createEmptySol();
        auto t0 = chrono::steady_clock::now();
        int i = 0;

        while (i < iterations)
        {
            auto now = chrono::steady_clock::now();
            auto ms = chrono::duration_cast<chrono::milliseconds>(now - t0).count();
            if (ms > max_time_ms_)
            {
                stopped_by_time = true;
                break;
            }

            constructiveHeuristic();
            localSearch();

            if (bestSol->cost > sol->cost)
            {
                bestSol = sol;
                iterations_to_best = i;

                if (ttt_mode_ && target_avg_value_ > 0.0 && time_to_target_ms < 0)
                {
                    if (bestSol->cost <= target_avg_value_)
                    {
                        time_to_target_ms = static_cast<long>(ms);
                        break;
                    }
                }
            }
            ++i;
        }

        total_iterations = i;
        auto tf = chrono::steady_clock::now();
        execution_time_ms =
            static_cast<long>(chrono::duration_cast<chrono::milliseconds>(tf - t0).count());
        return *bestSol;
    }

    int total_iterations{0};
    int iterations_to_best{0};
    long execution_time_ms{0};
    bool stopped_by_time{false};
    long time_to_target_ms{-1};

   private:
    long max_time_ms_;
    double target_avg_value_;
    bool ttt_mode_;
};

class GRASP_KMedoids_RW_WithStopping : public GRASP_KMedoids_RW
{
   public:
    GRASP_KMedoids_RW_WithStopping(double alpha, int iterations,
                                   vector<vector<double>>& D, int k,
                                   GRASP_KMedoids_RW::LSSearch mode, long max_time_ms,
                                   double target_avg_value = -1.0, bool ttt_mode = false)
        : GRASP_KMedoids_RW(alpha, iterations, D, k, mode),
          max_time_ms_(max_time_ms),
          target_avg_value_(target_avg_value),
          ttt_mode_(ttt_mode)
    {
    }

    Solution<int> solve()
    {
        bestSol = createEmptySol();
        auto t0 = chrono::steady_clock::now();
        int i = 0;
        while (i < iterations)
        {
            auto now = chrono::steady_clock::now();
            auto ms = chrono::duration_cast<chrono::milliseconds>(now - t0).count();
            if (ms > max_time_ms_)
            {
                stopped_by_time = true;
                break;
            }

            constructiveHeuristic();
            localSearch();

            if (bestSol->cost > sol->cost)
            {
                bestSol = sol;
                iterations_to_best = i;
                if (ttt_mode_ && target_avg_value_ > 0.0 && time_to_target_ms < 0)
                {
                    if (bestSol->cost <= target_avg_value_)
                    {
                        time_to_target_ms = (long) ms;
                        break;
                    }
                }
            }
            ++i;
        }
        total_iterations = i;
        auto tf = chrono::steady_clock::now();
        execution_time_ms =
            (long) chrono::duration_cast<chrono::milliseconds>(tf - t0).count();
        return *bestSol;
    }

    int total_iterations{0}, iterations_to_best{0};
    long execution_time_ms{0};
    bool stopped_by_time{false};
    long time_to_target_ms{-1};

   private:
    long max_time_ms_;
    double target_avg_value_;
    bool ttt_mode_;
};

class GRASP_KMedoids_FI_WithStopping : public GRASP_KMedoids_FI
{
   public:
    GRASP_KMedoids_FI_WithStopping(double alpha, int iterations,
                                   vector<vector<double>>& D, int k, long max_time_ms,
                                   double target_avg_value = -1.0, bool ttt_mode = false)
        : GRASP_KMedoids_FI(alpha, iterations, D, k),
          max_time_ms_(max_time_ms),
          target_avg_value_(target_avg_value),
          ttt_mode_(ttt_mode)
    {
    }

    Solution<int> solve()
    {
        bestSol = createEmptySol();
        auto t0 = chrono::steady_clock::now();
        int i = 0;

        while (i < iterations)
        {
            auto now = chrono::steady_clock::now();
            auto ms = chrono::duration_cast<chrono::milliseconds>(now - t0).count();
            if (ms > max_time_ms_)
            {
                stopped_by_time = true;
                break;
            }

            constructiveHeuristic();
            localSearch();

            if (bestSol->cost > sol->cost)
            {
                bestSol = sol;
                iterations_to_best = i;

                if (ttt_mode_ && target_avg_value_ > 0.0 && time_to_target_ms < 0)
                {
                    if (bestSol->cost <= target_avg_value_)
                    {
                        time_to_target_ms = static_cast<long>(ms);
                        break;
                    }
                }
            }
            ++i;
        }

        total_iterations = i;
        auto tf = chrono::steady_clock::now();
        execution_time_ms = static_cast<long>(
            chrono::duration_cast<chrono::milliseconds>(tf - t0).count());
        return *bestSol;
    }

    int total_iterations{0};
    int iterations_to_best{0};
    long execution_time_ms{0};
    bool stopped_by_time{false};
    long time_to_target_ms{-1};

   private:
    long max_time_ms_;
    double target_avg_value_;
    bool ttt_mode_;
};

class GRASP_KMedoids_RPG_WithStopping : public GRASP_KMedoids_RPG
{
   public:
    GRASP_KMedoids_RPG_WithStopping(double alpha, int iterations, vector<vector<double>>& D, int k,
                                    int p, long max_time_ms, double target_avg_value = -1.0,
                                    bool ttt_mode = false)
        : GRASP_KMedoids_RPG(alpha, iterations, D, k, p),
          max_time_ms_(max_time_ms),
          target_avg_value_(target_avg_value),
          ttt_mode_(ttt_mode)
    {
    }

    Solution<int> solve()
    {
        bestSol = createEmptySol();
        auto t0 = chrono::steady_clock::now();
        int i = 0;

        while (i < iterations)
        {
            auto now = chrono::steady_clock::now();
            auto ms = chrono::duration_cast<chrono::milliseconds>(now - t0).count();
            if (ms > max_time_ms_)
            {
                stopped_by_time = true;
                break;
            }

            constructiveHeuristic();
            localSearch();

            if (bestSol->cost > sol->cost)
            {
                bestSol = sol;
                iterations_to_best = i;

                if (ttt_mode_ && target_avg_value_ > 0.0 && time_to_target_ms < 0)
                {
                    if (bestSol->cost <= target_avg_value_)
                    {
                        time_to_target_ms = static_cast<long>(ms);
                        break;
                    }
                }
            }
            ++i;
        }

        total_iterations = i;
        auto tf = chrono::steady_clock::now();
        execution_time_ms =
            static_cast<long>(chrono::duration_cast<chrono::milliseconds>(tf - t0).count());
        return *bestSol;
    }

    int total_iterations{0};
    int iterations_to_best{0};
    long execution_time_ms{0};
    bool stopped_by_time{false};
    long time_to_target_ms{-1};

   private:
    long max_time_ms_;
    double target_avg_value_;
    bool ttt_mode_;
};

class GRASP_KMedoids_POP_WithStopping : public GRASP_KMedoids_POP
{
   public:
    GRASP_KMedoids_POP_WithStopping(double alpha, int iterations,
                                    vector<vector<double>>& D, int k, long max_time_ms,
                                    double target_avg_value = -1.0, bool ttt_mode = false)
        : GRASP_KMedoids_POP(alpha, iterations, D, k),
          max_time_ms_(max_time_ms),
          target_avg_value_(target_avg_value),
          ttt_mode_(ttt_mode)
    {
    }

    Solution<int> solve()
    {
        bestSol = createEmptySol();
        auto t0 = chrono::steady_clock::now();
        int i = 0;
        while (i < iterations)
        {
            auto now = chrono::steady_clock::now();
            auto ms = chrono::duration_cast<chrono::milliseconds>(now - t0).count();
            if (ms > max_time_ms_)
            {
                stopped_by_time = true;
                break;
            }

            constructiveHeuristic();
            localSearch();

            if (bestSol->cost > sol->cost)
            {
                bestSol = sol;
                iterations_to_best = i;
                if (ttt_mode_ && target_avg_value_ > 0.0 && time_to_target_ms < 0)
                {
                    if (bestSol->cost <= target_avg_value_)
                    {
                        time_to_target_ms = (long) ms;
                        break;
                    }
                }
            }
            ++i;
        }
        total_iterations = i;
        auto tf = chrono::steady_clock::now();
        execution_time_ms =
            (long) chrono::duration_cast<chrono::milliseconds>(tf - t0).count();
        return *bestSol;
    }

    int total_iterations{0}, iterations_to_best{0};
    long execution_time_ms{0};
    bool stopped_by_time{false};
    long time_to_target_ms{-1};

   private:
    long max_time_ms_;
    double target_avg_value_;
    bool ttt_mode_;
};

vector<ExperimentConfig> generate_configurations()
{
    vector<ExperimentConfig> cfgs;
    cfgs.reserve(ALPHA_VALUES.size() + P_VALUES.size());

    for (double a : ALPHA_VALUES)
        cfgs.push_back(
            ExperimentConfig{"GRASP_alpha=" + to_string(a), SolverKind::Standard, a, -1});

    for (double a : ALPHA_VALUES)
        cfgs.push_back(
            ExperimentConfig{"GRASP_RW_BI_alpha=" + to_string(a), SolverKind::RW_BI, a, -1});

    for (double a : ALPHA_VALUES)
        cfgs.push_back(
            ExperimentConfig{"GRASP_FI_alpha=" + to_string(a), SolverKind::StandardFI, a, -1});

    for (int p : P_VALUES)
        cfgs.push_back(ExperimentConfig{"RPG_p=" + to_string(p), SolverKind::RPG, -1.0, p});

    for (double a : ALPHA_VALUES)
        cfgs.push_back(ExperimentConfig{"GRASP_POP_alpha=" + to_string(a), SolverKind::POP, a, -1});

    return cfgs;
}

vector<vector<double>> load_distance_matrix(string& instance_path)
{
    auto X = load_i_dataset(instance_path, ';', ',');
    if (USE_ZSCORE) zscore_inplace(X, 1);
    return pairwise_euclidean(X);
}

void save_ttt_header_if_needed(string& csv_path)
{
    filesystem::create_directories(filesystem::path(csv_path).parent_path());
    bool exists = filesystem::exists(csv_path);
    ofstream f(csv_path, ios::out | ios::app);
    if (!f.is_open()) throw runtime_error("Failed to open TTT CSV: " + csv_path);
    if (!exists) f << "instance,file,k,config,target_avg,run_idx,time_to_target_ms\n";
}

void append_ttt_line(string& csv_path, string& instance_file, int k, string& config_name,
                     double target_avg, int run_idx, long ttt_ms)
{
    ofstream f(csv_path, ios::out | ios::app);
    if (!f.is_open()) throw runtime_error("Failed to open TTT CSV: " + csv_path);
    f.setf(ios::fixed);
    f << INSTANCES_DIR << '/' << instance_file << ',' << instance_file << ',' << k << ','
      << config_name << ',' << setprecision(6) << target_avg << ',' << run_idx << ',' << ttt_ms
      << '\n';
}

ExperimentResult run_experiment(ExperimentConfig& config, string& instance_file, int k,
                                string& ttt_csv)
{
    cout << "  Running: " << config.name << " | k=" << k << " | on " << instance_file << "\n";

    string instance_path = INSTANCES_DIR + "/" + instance_file;
    auto D = load_distance_matrix(instance_path);
    int n = static_cast<int>(D.size());

    if (ENABLE_TTT_MODE)
    {
        double target_avg = get_target_avg_for(instance_file, k);
        if (!(target_avg > 0.0))
        {
            cout << "    [WARN] target_avg não definido (>0). Os tempos ficarão como -1.\n";
        }

        save_ttt_header_if_needed(ttt_csv);

        for (int run = 0; run < TTT_RUNS; ++run)
        {
            if (config.kind == SolverKind::Standard)
            {
                GRASP_KMedoids_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                                  MAX_TIME_MILLIS, target_avg, true);
                auto sol = grasp.solve();
                append_ttt_line(ttt_csv, instance_file, k, config.name, target_avg, run,
                                grasp.time_to_target_ms);
            }
            else if (config.kind == SolverKind::StandardFI)
            {
                GRASP_KMedoids_FI_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                                     MAX_TIME_MILLIS, target_avg, true);
                auto sol = grasp.solve();
                append_ttt_line(ttt_csv, instance_file, k, config.name, target_avg, run,
                                grasp.time_to_target_ms);
            }
            else if (config.kind == SolverKind::POP)
            {
                GRASP_KMedoids_POP_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                                      MAX_TIME_MILLIS, target_avg, true);
                auto sol = grasp.solve();
                append_ttt_line(ttt_csv, instance_file, k, config.name, target_avg, run,
                                grasp.time_to_target_ms);
            }
            else if (config.kind == SolverKind::RW_BI)
            {
                GRASP_KMedoids_RW_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                                     GRASP_KMedoids_RW::LSSearch::BestImproving,
                                                     MAX_TIME_MILLIS, target_avg, true);
                auto sol = grasp.solve();
                append_ttt_line(ttt_csv, instance_file, k, config.name, target_avg, run,
                                grasp.time_to_target_ms);
            }
            else
            {
                GRASP_KMedoids_RPG_WithStopping grasp(0.0, MAX_TOTAL_ITERATIONS, D, k, config.p,
                                                      MAX_TIME_MILLIS, target_avg, true);
                auto sol = grasp.solve();
                append_ttt_line(ttt_csv, instance_file, k, config.name, target_avg, run,
                                grasp.time_to_target_ms);
            }
        }

        ExperimentResult r{};
        r.config = config.name;
        r.file = instance_file;
        r.n = n;
        r.k = k;
        r.alpha = (config.kind == SolverKind::RPG) ? -1.0 : config.alpha;

        r.construct_mode = (config.kind == SolverKind::RPG) ? "RPG"
            : (config.kind == SolverKind::POP)              ? "STANDARD+POP"
                                                            : "STANDARD";

        r.ls_mode = (config.kind == SolverKind::StandardFI) ? "FIRST_IMPROVING"
            : (config.kind == SolverKind::RW_BI)            ? "FAST_INTERCHANGE_BI"
                                                            : "BEST_IMPROVING";
        r.iterations = MAX_TOTAL_ITERATIONS;
        r.time_limit_s = static_cast<double>(MAX_TIME_MILLIS) / 1000.0;
        r.timed_out = false;
        r.max_value = 0.0;
        r.size = 0;
        r.feasible = true;
        r.time_s = 0.0;
        r.elements = "";
        return r;
    }

    Solution<int> sol;
    int total_iterations = 0, iterations_to_best = 0;
    long exec_ms = 0;
    bool stopped_by_time = false;

    if (config.kind == SolverKind::Standard)
    {
        GRASP_KMedoids_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                          MAX_TIME_MILLIS);
        sol = grasp.solve();
        total_iterations = grasp.total_iterations;
        iterations_to_best = grasp.iterations_to_best;
        exec_ms = grasp.execution_time_ms;
        stopped_by_time = grasp.stopped_by_time;
    }
    else if (config.kind == SolverKind::StandardFI)
    {
        GRASP_KMedoids_FI_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                             MAX_TIME_MILLIS);
        sol = grasp.solve();
        total_iterations = grasp.total_iterations;
        iterations_to_best = grasp.iterations_to_best;
        exec_ms = grasp.execution_time_ms;
        stopped_by_time = grasp.stopped_by_time;
    }
    else if (config.kind == SolverKind::POP)
    {
        GRASP_KMedoids_POP_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                              MAX_TIME_MILLIS);
        sol = grasp.solve();
        total_iterations = grasp.total_iterations;
        iterations_to_best = grasp.iterations_to_best;
        exec_ms = grasp.execution_time_ms;
        stopped_by_time = grasp.stopped_by_time;
    }
    else if (config.kind == SolverKind::RW_BI)
    {
        GRASP_KMedoids_RW_WithStopping grasp(config.alpha, MAX_TOTAL_ITERATIONS, D, k,
                                             GRASP_KMedoids_RW::LSSearch::BestImproving,
                                             MAX_TIME_MILLIS);
        sol = grasp.solve();
        total_iterations = grasp.total_iterations;
        iterations_to_best = grasp.iterations_to_best;
        exec_ms = grasp.execution_time_ms;
        stopped_by_time = grasp.stopped_by_time;
    }
    else
    {
        GRASP_KMedoids_RPG_WithStopping grasp(0.0, MAX_TOTAL_ITERATIONS, D, k, config.p,
                                              MAX_TIME_MILLIS);
        sol = grasp.solve();
        total_iterations = grasp.total_iterations;
        iterations_to_best = grasp.iterations_to_best;
        exec_ms = grasp.execution_time_ms;
        stopped_by_time = grasp.stopped_by_time;
    }

    double best_avg = sol.cost;
    double best_total = best_avg * static_cast<double>(n);
    double time_sec = static_cast<double>(exec_ms) / 1000.0;

    cout << "    -> Total: " << fixed << setprecision(6) << best_total << " | Avg: " << best_avg
         << ", Iter: " << total_iterations << ", Time: " << setprecision(3) << time_sec << "s\n";

    ostringstream els;
    for (size_t i = 0; i < sol.size(); ++i)
    {
        if (i) els << ' ';
        els << sol[i];
    }

    ExperimentResult r;
    r.config = config.name;
    r.file = instance_file;
    r.n = n;
    r.k = k;
    r.alpha = (config.kind == SolverKind::RPG) ? -1.0 : config.alpha;

    r.construct_mode = (config.kind == SolverKind::RPG) ? "RPG"
        : (config.kind == SolverKind::POP)              ? "STANDARD+POP"
                                                        : "STANDARD";

    r.ls_mode = (config.kind == SolverKind::StandardFI) ? "FIRST_IMPROVING"
        : (config.kind == SolverKind::RW_BI)            ? "FAST_INTERCHANGE_BI"
                                                        : "BEST_IMPROVING";
    r.reactive_alphas = "";
    r.reactive_block = "";
    r.sample_size = (config.kind == SolverKind::RPG) ? to_string(config.p) : "";
    r.iterations = MAX_TOTAL_ITERATIONS;
    r.time_limit_s = static_cast<double>(MAX_TIME_MILLIS) / 1000.0;
    r.timed_out = stopped_by_time;
    r.max_value = -best_avg;
    r.size = static_cast<int>(sol.size());
    r.feasible = true;
    r.time_s = time_sec;
    r.elements = els.str();

    return r;
}

bool USE_FILTERS = true;

vector<string> ALLOW_INSTANCES = {
                                //   "2-FACE.i",
                                //   "200DATA.I",
                                //   "400p3c.i",
                                  "A1.I",
                                  "BreastB.I",
                                  "Compound.I",
                                  "Concrete_Data.i",
                                  "DBLCA.I",
                                  "DBLCB.I",
                                  "DOWJONES.I",
                                  "Normal300.i",
                                  "PARKINSONS.i",
                                  "Prima_Indians.i",
                                  "SPRDESP.I",
                                  "TRIPADVISOR.I",
                                  "Uniform400.i",
                                  "Uniform700.i",
                                  "aggregation.I",
                                  "banknote_authentication.I",
                                  "broken-ring.i",
                                  "bupa.I",
                                  "chart.i",
                                  "ecoli.i",
                                  "face.i",
                                  "forestfires.I",
                                  "gauss9.i",
                                  "glass.I",
                                  "haberman.i",
                                  "hayes-roth.I",
                                  "indian.i",
                                  "indochina_combat.I",
                                  "ionosphere.I",
                                  "iris.i",
                                  "maronna.i",
                                  "moreshapes.i",
                                  "new-thyroid.I",
                                  "numbers2.i",
                                  "outliers.i",
                                  "pib100.i",
                                  "ruspini.i",
                                  "sonar.I",
                                  "spherical_4d3c.i",
                                  "synthetic_control.i",
                                  "vowel2.i",
                                  "wdbc.I",
                                  "wine.i",
                                  "yeast.i",
                                  "waveform.I"};

vector<int> ALLOW_KS = {3, 4, 5, 6};

// vector<string> ALLOW_CONFIG_PREFIX = {"GRASP_alpha=0.05", "RPG_p=10"};
// vector<string> ALLOW_CONFIG_PREFIX = {"GRASP_alpha=0.05"};
vector<string> ALLOW_CONFIG_PREFIX = {
    "GRASP_alpha=", "GRASP_FI_alpha=", "GRASP_POP_alpha=", "GRASP_RW_BI_alpha=", "RPG_p="};

vector<string> BLOCK_CONFIG_PREFIX = {};

bool starts_with(const string& s, const string& p) { return s.rfind(p, 0) == 0; }

bool contains_str(const vector<string>& v, const string& x)
{
    for (auto& s : v)
        if (s == x) return true;
    return false;
}

bool contains_int(const vector<int>& v, int x)
{
    for (auto& a : v)
        if (a == x) return true;
    return false;
}

bool match_any_prefix(const string& s, const vector<string>& prefixes)
{
    for (auto& p : prefixes)
        if (starts_with(s, p)) return true;
    return false;
}

bool should_run_config(const string& inst, int k, const ExperimentConfig& cfg)
{
    if (!USE_FILTERS) return true;
    if (!ALLOW_INSTANCES.empty() && !contains_str(ALLOW_INSTANCES, inst)) return false;
    if (!ALLOW_KS.empty() && !contains_int(ALLOW_KS, k)) return false;
    if (!BLOCK_CONFIG_PREFIX.empty() && match_any_prefix(cfg.name, BLOCK_CONFIG_PREFIX))
        return false;
    if (!ALLOW_CONFIG_PREFIX.empty() && !match_any_prefix(cfg.name, ALLOW_CONFIG_PREFIX))
        return false;
    return true;
}

void save_results_to_csv(vector<ExperimentResult>& results, string& output_file)
{
    filesystem::create_directories(filesystem::path(output_file).parent_path());

    ofstream f(output_file, ios::out | ios::trunc);
    if (!f.is_open()) throw runtime_error("Failed to open output CSV: " + output_file);

    f << "config,file,n,k,alpha,construct_mode,ls_mode,reactive_alphas,reactive_block,"
         "sample_size,iterations,time_limit_s,timed_out,max_value,size,feasible,time_s,elements\n";

    f.setf(ios::fixed);
    for (const auto& r : results)
    {
        f << r.config << ',' << r.file << ',' << r.n << ',' << r.k << ',' << setprecision(4)
          << r.alpha << ',' << r.construct_mode << ',' << r.ls_mode << ",\"" << r.reactive_alphas
          << "\"," << r.reactive_block << ',' << r.sample_size << ',' << r.iterations << ','
          << setprecision(0) << r.time_limit_s << ',' << (r.timed_out ? "true" : "false") << ','
          << setprecision(6) << r.max_value << ',' << r.size << ','
          << (r.feasible ? "true" : "false") << ',' << setprecision(3) << r.time_s << ",\""
          << r.elements << "\"\n";
    }
    cout << "\nResults saved to: " << output_file << "\n";
}

void print_summary_statistics(vector<ExperimentResult>& results)
{
    cout << "\n=== SUMMARY STATISTICS ===\n\n";

    vector<pair<string, int>> keys;
    for (auto& r : results) keys.emplace_back(r.file, r.k);
    sort(keys.begin(), keys.end());
    keys.erase(unique(keys.begin(), keys.end()), keys.end());

    for (auto [inst, k] : keys)
    {
        vector<ExperimentResult> subset;
        for (auto& r : results)
            if (r.file == inst && r.k == k) subset.push_back(r);

        if (subset.empty()) continue;

        auto best_it = min_element(subset.begin(), subset.end(),
                                   [](const ExperimentResult& a, const ExperimentResult& b)
                                   { return a.max_value > b.max_value; });

        auto& best = *best_it;
        cout << "Instance: " << inst << " | k=" << k << "\n";
        cout << "  Best configuration: " << best.config << "\n";
        cout << "  Best avg cost: " << fixed << setprecision(6) << (-best.max_value) << "\n";
        cout << "  Time: " << setprecision(3) << best.time_s << " seconds\n\n";
    }
}

int main()
{
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    tm tm{};

#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

    string output_csv = RESULTS_DIR + "/grasp_kmedoids_" + string(buf) + ".csv";
    string ttt_csv = RESULTS_DIR + "/ttt_grasp_kmedoids_" + string(buf) + ".csv";

    cout << "==============================================\n";
    cout << "GRASP K-MEDOIDS\n";
    cout << "==============================================\n";
    cout << "Tempo máximo por rodada: " << (MAX_TIME_MILLIS / 1000) << " seconds\n";
    cout << "Arquivo de saída: " << output_csv << "\n";
    if (ENABLE_TTT_MODE)
    {
        cout << "TTT MODE: ON\n";
        cout << "  -> runs: " << TTT_RUNS << "\n";
        cout << "  -> targets: por instância x k\n";
        cout << "  -> ttt_csv: " << ttt_csv << "\n";
    }
    cout << "==============================================\n\n";

    auto configs = generate_configurations();
    vector<ExperimentResult> all_results;

    cout << "Total configurações: " << configs.size() << "\n";
    cout << "Total instâncias: " << INSTANCE_FILES.size() << "\n";
    cout << "Total experimentos: " << (configs.size() * INSTANCE_FILES.size() * K_VALUES.size())
         << "\n\n";
    cout << "Start...\n\n";

    auto start = chrono::steady_clock::now();
    int exp_count = 0;

    for (auto& inst : INSTANCE_FILES)
    {
        cout << "\n--- Instância: " << inst << " ---\n";
        for (int k : K_VALUES)
        {
            for (auto& cfg : configs)
            {
                ++exp_count;

                if (!should_run_config(inst, k, cfg))
                {
                    cout << "  [skip] " << cfg.name << " | k=" << k << " | on " << inst << "\n";
                    continue;
                }

                cout << "\n[Experimento " << exp_count << "/"
                     << (configs.size() * INSTANCE_FILES.size() * K_VALUES.size()) << "]\n";

                auto res = run_experiment(cfg, inst, k, ttt_csv);
                if (!ENABLE_TTT_MODE)
                {
                    all_results.push_back(res);

                    save_results_to_csv(all_results, output_csv);
                }
            }
        }
    }

    if (!ENABLE_TTT_MODE)
    {
        save_results_to_csv(all_results, output_csv);
    }
    else
    {
        cout << "\nTTT results saved to: " << ttt_csv << "\n";
    }

    auto end = chrono::steady_clock::now();
    double total_time_min = chrono::duration_cast<chrono::seconds>(end - start).count() / 60.0;

    cout << "Experimentos completos\n";
    cout << "Tempo Total: " << fixed << setprecision(2) << total_time_min << " minutos\n";
    if (!ENABLE_TTT_MODE)
    {
        cout << "Resultados salvos: " << output_csv << "\n";
        print_summary_statistics(all_results);
    }
}
