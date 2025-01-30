# Disco C++

🔥 Recommendations for C++ using collaborative filtering

- Supports user-based and item-based recommendations
- Works with explicit and implicit feedback
- Uses high-performance matrix factorization

🎉 Zero dependencies

[![Build Status](https://github.com/ankane/disco-cpp/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/disco-cpp/actions)

## Installation

Add [the header](https://raw.githubusercontent.com/ankane/disco-cpp/v0.1.2/include/disco.hpp) to your project (supports C++20 and greater).

There is also support for CMake and FetchContent:

```cmake
include(FetchContent)

FetchContent_Declare(disco GIT_REPOSITORY https://github.com/ankane/disco-cpp.git GIT_TAG v0.1.2)
FetchContent_MakeAvailable(disco)

target_link_libraries(app PRIVATE disco::disco)
```

## Getting Started

Include the header

```cpp
#include "disco.hpp"
```

Prep your data in the format `user_id, item_id, value`

```cpp
using disco::Dataset;

auto data = new Dataset<std::string, std::string>();
data.push("user_a", "item_a", 5.0);
data.push("user_a", "item_b", 3.5);
data.push("user_b", "item_a", 4.0);
```

IDs can be integers, strings, or any other hashable data type

```cpp
data.push(1, "item_a", 5.0);
```

If users rate items directly, this is known as explicit feedback. Fit the recommender with:

```cpp
using disco::Recommender;

auto recommender = Recommender<std::string, std::string>::fit_explicit(data);
```

If users don’t rate items directly (for instance, they’re purchasing items or reading posts), this is known as implicit feedback. Use `1.0` or a value like number of purchases or page views for the dataset, and fit the recommender with:

```cpp
auto recommender = Recommender<std::string, std::string>::fit_implicit(data);
```

Get user-based recommendations - “users like you also liked”

```cpp
recommender.user_recs(user_id);
```

Get item-based recommendations - “users who liked this item also liked”

```cpp
recommender.item_recs(item_id);
```

Use the `count` option to specify the number of recommendations (default is 5)

```cpp
recommender.user_recs(user_id, 5);
```

Get predicted ratings for a specific user and item

```cpp
recommender.predict(user_id, item_id);
```

Get similar users

```cpp
recommender.similar_users(user_id);
```

## Examples

### MovieLens

Download the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

And use:

```cpp
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "disco.hpp"

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
```

## Storing Recommendations

Save recommendations to your database.

Alternatively, you can store only the factors and use a library like [pgvector-cpp](https://github.com/pgvector/pgvector-cpp). See an [example](https://github.com/pgvector/pgvector-cpp/blob/master/examples/disco/example.cpp).

## Algorithms

Disco uses high-performance matrix factorization.

- For explicit feedback, it uses the [stochastic gradient method with twin learners](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf)
- For implicit feedback, it uses the [conjugate gradient method](https://www.benfrederickson.com/fast-implicit-matrix-factorization/)

Specify the number of factors and iterations

```cpp
auto recommender = Recommender<int, int>::fit_explicit(data, { .factors = 8, .iterations = 20 });
```

## Progress

Pass a callback to show progress

```cpp
auto callback = [](const disco::FitInfo& info) {
    std::cout << info.iteration << ": " << info.train_loss << std::endl;
};
auto recommender = Recommender<int, int>::fit_explicit(data, { .callback = callback });
```

Note: `train_loss` is not available for implicit feedback

## Cold Start

Collaborative filtering suffers from the [cold start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)). It’s unable to make good recommendations without data on a user or item, which is problematic for new users and items.

```cpp
recommender.user_recs(new_user_id, 5); // returns empty array
```

There are a number of ways to deal with this, but here are some common ones:

- For user-based recommendations, show new users the most popular items
- For item-based recommendations, make content-based recommendations

## Reference

Get ids

```cpp
recommender.user_ids();
recommender.item_ids();
```

Get the global mean

```cpp
recommender.global_mean();
```

Get factors

```cpp
recommender.user_factors(user_id);
recommender.item_factors(item_id);
```

## References

- [A Learning-rate Schedule for Stochastic Gradient Methods to Matrix Factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf)
- [Faster Implicit Matrix Factorization](https://www.benfrederickson.com/fast-implicit-matrix-factorization/)

## History

View the [changelog](https://github.com/ankane/disco-cpp/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/disco-cpp/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/disco-cpp/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/disco-cpp.git
cd disco-cpp
g++ -std=c++20 -Wall -Wextra -Werror -o test/main test/main.cpp
test/main
```
