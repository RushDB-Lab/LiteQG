#include "../symqglib/utils/lazy_cleanup_set.hpp"
#include <unordered_set>
#include <iostream>
#include <chrono>

void lab1(int turn, int number)
{
  using namespace std::chrono;

  auto start = high_resolution_clock::now();

  std::unordered_set<int> set(100);

  for (int i = 0; i < turn; i++)
  {
    for (int j = 0; j < number; j++)
    {
      set.emplace(j);
    }
    for (int j = 0; j < number; j++)
    {
      set.find(j) == set.end();
    }
    set.clear();
  }

  auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
  std::cout << "std::unordered_set time take: " << duration.count() << " ms" << std::endl;
}

void lab2(int turn, int number)
{
  using namespace std::chrono;

  auto start = high_resolution_clock::now();

  symqg::LazyCleanupSet<int> set(100);

  for (int i = 0; i < turn; i++)
  {
    for (int j = 0; j < number; j++)
    {
      set.emplace(j);
    }
    for (int j = 0; j < number; j++)
    {
      set.contains(j);
    }
    set.clear();
  }

  auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
  std::cout << "symqg::LazyCleanupSet time take: " << duration.count() << " ms" << std::endl;
}

int main()
{
  for (int i = 100; i < 1e7; i *= 10)
  {
    std::cout << std::endl
              << "number: " << i << std::endl;
    lab1(1000, i);
    lab2(1000, i);
  }

  std::cout << std::endl;
}
