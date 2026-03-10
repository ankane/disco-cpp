#include <algorithm>
#include <cassert>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "../include/disco.hpp"

using disco::Dataset;
using disco::Recommender;

template<typename T> void assert_eq(const std::vector<T>& a, const std::vector<T>& b) {
    assert(std::equal(a.begin(), a.end(), b.begin(), b.end()));
}

template<typename T> void assert_eq(std::span<const T> a, const std::vector<T>& b) {
    assert(std::equal(a.begin(), a.end(), b.begin(), b.end()));
}

std::optional<Dataset<int, std::string>> load_movielens() {
    // https://grouplens.org/datasets/movielens/100k/
    const char* path = std::getenv("MOVIELENS_100K_PATH");
    if (!path) {
        return std::nullopt;
    }

    std::string line;

    // read movies
    std::unordered_map<std::string, std::string> movies;
    std::ifstream movies_file(std::string{path} + "/u.item");
    assert(movies_file.is_open());
    while (std::getline(movies_file, line)) {
        std::string::size_type n = line.find('|');
        std::string::size_type n2 = line.find('|', n + 1);
        movies.emplace(line.substr(0, n), line.substr(n + 1, n2 - n - 1));
    }

    // read ratings and create dataset
    Dataset<int, std::string> data;
    std::ifstream ratings_file(std::string{path} + "/u.data");
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

void test_explicit() {
    auto data = load_movielens();
    if (!data) {
        return;
    }

    auto recommender = Recommender<int, std::string>::fit_explicit(*data, { .factors = 20 });
    auto recs = recommender.item_recs("Star Wars (1977)");
    assert(recs.size() == 5);
}

void test_implicit() {
    auto data = load_movielens();
    if (!data) {
        return;
    }

    auto recommender = Recommender<int, std::string>::fit_implicit(*data, { .factors = 20 });
    auto recs = recommender.item_recs("Star Wars (1977)");
    assert(recs.size() == 5);
}

void test_rated() {
    Dataset<int, std::string> data;
    data.push(1, "A", 1.0);
    data.push(1, "B", 1.0);
    data.push(1, "C", 1.0);
    data.push(1, "D", 1.0);
    data.push(2, "C", 1.0);
    data.push(2, "D", 1.0);
    data.push(2, "E", 1.0);
    data.push(2, "F", 1.0);

    auto recommender = Recommender<int, std::string>::fit_implicit(data);

    std::vector<std::string> item_ids;
    for (const auto& v : recommender.user_recs(1, 5)) {
        item_ids.push_back(v.first);
    }
    std::ranges::sort(item_ids);
    assert_eq(item_ids, {"E", "F"});

    item_ids.clear();
    for (const auto& v : recommender.user_recs(2, 5)) {
        item_ids.push_back(v.first);
    }
    std::ranges::sort(item_ids);
    assert_eq(item_ids, {"A", "B"});
}

void test_item_recs_same_score() {
    Dataset<int, std::string> data;
    data.push(1, "A", 1.0);
    data.push(1, "B", 1.0);
    data.push(2, "C", 1.0);

    auto recommender = Recommender<int, std::string>::fit_implicit(data);
    std::vector<std::string> item_ids;
    for (const auto& v : recommender.item_recs("A", 5)) {
        item_ids.push_back(v.first);
    }
    assert_eq(item_ids, {"B", "C"});
}

void test_ids() {
    Dataset<int, std::string> data;
    data.push(1, "A", 1.0);
    data.push(1, "B", 1.0);
    data.push(2, "B", 1.0);

    auto recommender = Recommender<int, std::string>::fit_implicit(data);
    assert_eq(recommender.user_ids(), {1, 2});
    assert_eq(recommender.item_ids(), {"A", "B"});
}

void test_factors() {
    Dataset<int, std::string> data;
    data.push(1, "A", 1.0);
    data.push(1, "B", 1.0);
    data.push(2, "B", 1.0);

    auto recommender = Recommender<int, std::string>::fit_implicit(data, { .factors = 20 });

    assert((*recommender.user_factors(1)).size() == 20);
    assert((*recommender.item_factors("A")).size() == 20);

    assert(recommender.user_factors(3) == std::nullopt);
    assert(recommender.item_factors("C") == std::nullopt);
}

void test_callback_explicit() {
    Dataset<int, std::string> data;
    data.push(1, "A", 1.0);

    size_t calls = 0;
    auto callback = [&calls](const disco::FitInfo& info) {
        assert(info.iteration == calls + 1);
        assert(!std::isnan(info.train_loss));
        calls++;
    };
    Recommender<int, std::string>::fit_explicit(data, { .callback = callback });
    assert(calls == 20);
}

void test_callback_implicit() {
    Dataset<int, std::string> data;
    data.push(1, "A", 1.0);

    size_t calls = 0;
    auto callback = [&calls](const disco::FitInfo& info) {
        assert(info.iteration == calls + 1);
        assert(std::isnan(info.train_loss));
        calls++;
    };
    Recommender<int, std::string>::fit_implicit(data, { .callback = callback });
    assert(calls == 20);
}

int main() {
    test_explicit();
    test_implicit();
    test_rated();
    test_item_recs_same_score();
    test_ids();
    test_factors();
    test_callback_explicit();
    test_callback_implicit();
    return 0;
}
