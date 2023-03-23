#include <iostream>
#include <string>

#include "grammar.h"

template <typename Rule> struct calc_action : tao::pegtl::nothing<Rule> {};

template <> struct calc_action<grammar::start> {
  template <typename ActionInput>
  static void apply(const ActionInput &in, std::string &out) {
    std::cout << "start rule" << std::endl;
  }
};

int main() {
  std::string out;
  tao::pegtl::memory_input<> in("1 + 3 * 3", "");
  tao::pegtl::parse<grammar::start, calc_action>(in, out);
  return 0;
}
