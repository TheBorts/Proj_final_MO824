#include <string>
#include <vector>

using namespace std;

double to_float(string token, char decimal = ',');

vector<vector<double>> load_i_dataset(const string& path, char sep = ';', char decimal = ',');

vector<vector<double>>& zscore_inplace(vector<vector<double>>& X, int ddof = 1);

vector<vector<double>> pairwise_euclidean(const vector<vector<double>>& X);
