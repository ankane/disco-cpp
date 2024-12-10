#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "../include/disco.hpp"

using disco::Dataset;
using disco::Recommender;

Dataset<int, std::string> load_movielens(const std::string& path) {
    std::string line;

    // read movies
    std::unordered_map<std::string, std::string> movies;
    std::ifstream movies_file(path + "/u.item");
    assert(movies_file.is_open());
    while (std::getline(movies_file, line)) {
        std::string::size_type n = line.find('|');
        std::string::size_type n2 = line.find('|', n + 1);
        movies.emplace(std::make_pair(line.substr(0, n), line.substr(n + 1, n2 - n - 1)));
    }

    // read ratings and create dataset
    auto data = Dataset<int, std::string>();
    std::ifstream ratings_file(path + "/u.data");
    assert(ratings_file.is_open());
    while (std::getline(ratings_file, line)) {
        std::string::size_type n = line.find('\t');
        std::string::size_type n2 = line.find('\t', n + 1);
        std::string::size_type n3 = line.find('\t', n2 + 1);
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
    char *movielens_path = std::getenv("MOVIELENS_100K_PATH");
    if (!movielens_path) {
        std::cout << "Set MOVIELENS_100K_PATH" << std::endl;
        return 1;
    }

    auto data = load_movielens(movielens_path);
    auto recommender = Recommender<int, std::string>::fit_explicit(data, { .factors = 20 });

    std::string movie = "Star Wars (1977)";
    std::cout << "Item-based recommendations for " << movie << std::endl;
    for (auto& rec : recommender.item_recs(movie)) {
        std::cout << "- " << rec.first << std::endl;
    }

    int user_id = 123;
    std::cout << std::endl << "User-based recommendations for " << user_id << std::endl;
    for (auto& rec : recommender.user_recs(user_id)) {
        std::cout << "- " << rec.first << std::endl;
    }

    return 0;
}
