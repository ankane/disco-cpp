#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "../include/disco.hpp"

using disco::Dataset;
using disco::Recommender;

Dataset<int, std::string> load_movielens(const std::string& path) {
    // read movies
    std::unordered_map<std::string, std::string> movies;
    std::ifstream movies_file{path + "/u.item"};
    assert(movies_file.is_open());
    std::string line;
    while (std::getline(movies_file, line)) {
        size_t n = line.find('|');
        size_t n2 = line.find('|', n + 1);
        movies.emplace(line.substr(0, n), line.substr(n + 1, n2 - n - 1));
    }

    // read ratings and create dataset
    Dataset<int, std::string> data;
    std::ifstream ratings_file{path + "/u.data"};
    assert(ratings_file.is_open());
    while (std::getline(ratings_file, line)) {
        size_t n = line.find('\t');
        size_t n2 = line.find('\t', n + 1);
        size_t n3 = line.find('\t', n2 + 1);
        data.push(
            std::stoi(line.substr(0, n)),
            movies.at(line.substr(n + 1, n2 - n - 1)),
            std::stof(line.substr(n2 + 1, n3 - n2 - 1))
        );
    }

    return data;
}

int main() {
    // https://grouplens.org/datasets/movielens/100k/
    const char* movielens_path = std::getenv("MOVIELENS_100K_PATH");
    if (!movielens_path) {
        std::cout << "Set MOVIELENS_100K_PATH" << std::endl;
        return 1;
    }

    Dataset<int, std::string> data = load_movielens(movielens_path);
    auto recommender = Recommender<int, std::string>::fit_explicit(data, { .factors = 20 });

    std::string movie{"Star Wars (1977)"};
    std::cout << "Item-based recommendations for " << movie << std::endl;
    for (const auto& [item_id, score] : recommender.item_recs(movie)) {
        std::cout << "- " << item_id << std::endl;
    }

    int user_id = 123;
    std::cout << std::endl << "User-based recommendations for " << user_id << std::endl;
    for (const auto& [item_id, score] : recommender.user_recs(user_id)) {
        std::cout << "- " << item_id << std::endl;
    }

    return 0;
}
