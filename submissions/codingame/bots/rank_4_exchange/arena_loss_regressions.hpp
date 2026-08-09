#pragma once

#include <array>
#include <string_view>

namespace papersoccer::rank_4_exchange_regressions {
struct Case { std::string_view label; int player_id; std::string_view prefix; };
inline constexpr std::array<Case, 13> kCases{{
  {"v39 opponent-opening-1 loss family", 1, "1"},
  {"v40 player-0 0/6/1 loss family", 0, "0/6"},
  {"opening-0 reply", 1, "0"},
  {"jacek family 0/0", 0, "0/0"},
  {"jacek family 0/0/1", 1, "0/0/1"},
  {"jacek family 0/0/3", 1, "0/0/3"},
  {"Deltaspace family 0/6/1", 1, "0/6/1"},
  {"Deltaspace family 0/6/5", 1, "0/6/5"},
  {"Deltaspace 0633 branch", 0, "0/6/5/4/5/53/61/0633"},
  {"Deltaspace 431 branch", 0, "0/6/5/4/5/53/61/431"},
  {"Deltaspace opening-5 branch", 0, "0/6/5/5"},
  {"Deltaspace player-1 003 branch", 1, "1/1/7/6/0/75/74/3/003"},
  {"jacek 5330 branch", 1, "0/0/3/67/27/45/5/2/5/6143/5/717271/1/7/5330"},
}};
}  // namespace papersoccer::rank_4_exchange_regressions
