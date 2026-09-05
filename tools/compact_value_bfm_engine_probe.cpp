// Source-bound diagnostic. Version comparisons are not pure speed A/B tests;
// identical-model macro variants can additionally compare their fixed traces.
#define COMPACT_VALUE_BFM_NO_MAIN
#ifndef COMPACT_ENGINE_SOURCE
#error "COMPACT_ENGINE_SOURCE must identify the exact standalone source"
#endif
#include COMPACT_ENGINE_SOURCE
#include <bit>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cv = compact_value_bfm;
using Clock = std::chrono::steady_clock;
struct Row {
  std::string id, action, trace;
  double ms{};
  std::uint64_t generated{}, evaluated{}, nodes{}, expansions{};
};
std::string trace(const cv::SearchResult &result) {
  std::ostringstream out;
  const auto &s=result.stats;
  out << std::bit_cast<std::uint32_t>(result.value) << ':' << result.solved;
  for (auto value : {s.expansions,s.generated_children,s.evaluated_children,
      s.tactical_children,s.cache_probes,s.cache_hits,s.cache_misses,
      s.widening_probes,s.widening_restrictions,s.widening_eligible,s.widening_deferred,
      s.reuse_probes,s.reuse_hits,s.reuse_misses,s.reuse_rejections,s.reused_children,
      s.generator_partial_paths,s.generator_proof_paths,s.duplicate_boundaries,
      s.fifo_extractions,s.lifo_extractions,static_cast<std::uint64_t>(s.tree_nodes),
      static_cast<std::uint64_t>(s.max_depth),static_cast<std::uint64_t>(s.deadline_reached),
      static_cast<std::uint64_t>(s.tree_cap_reached),static_cast<std::uint64_t>(s.expansion_cap_reached)})
    out << ':' << value;
  for (const auto &root:result.root_actions)
    out << ';' << root.action.text() << ':' << static_cast<int>(root.tactical)
        << ':' << std::bit_cast<std::uint32_t>(root.value)
        << ':' << std::bit_cast<std::uint32_t>(root.initial_value)
        << ':' << root.visits << ':' << root.selection_visits << ':' << root.solved << ':' << root.order;
  return out.str();
}
cv::Action action(const std::string &text) {
  cv::Action result;
  for (char value:text) {
    if (value<'0' || value>'7' || result.length==result.directions.size())
      throw std::invalid_argument("invalid action");
    result.directions[result.length++]=static_cast<std::uint8_t>(value-'0');
  }
  return result;
}
int main(int argc,char **argv) {
  try {
    if (argc!=3) throw std::invalid_argument("usage: probe ROOTS_TSV fixed|clock");
    const std::string mode=argv[2];
    if (mode!="fixed" && mode!="clock") throw std::invalid_argument("invalid mode");
    std::ifstream input(argv[1]);
    if (!input) throw std::invalid_argument("missing roots");
    std::vector<Row> rows;
    std::string line;
    while (std::getline(input,line)) {
      if (line=="root_id\ttranscript") continue;
      const auto tab=line.find('\t');
      if (tab==std::string::npos) throw std::invalid_argument("invalid row");
      cv::State state=cv::initial_state();
      std::istringstream stream(line.substr(tab+1));std::string turn;
      while (std::getline(stream,turn,'/')) {
        if (!cv::apply_action(state,action(turn))) throw std::invalid_argument("illegal root");
      }
      if (state.terminal() || state.ply<8) throw std::invalid_argument("root must have at least eight edges");
      cv::SearchConfig config;
      config.max_actions=250;config.root_partial_paths=4000;config.nonroot_partial_paths=512;
      config.max_tree_nodes=mode=="fixed" ? 4000 : 80000;
      config.max_expansions=mode=="fixed" ? 1000 : 2000000;
      config.exploration=.95;config.fpu=.5;config.final_visit_weight=1;
      const auto started=Clock::now();
      const auto emergency=cv::emergency_complete_action(state);
      const auto deadline=mode=="fixed" ? Clock::time_point::max()
          : started+std::chrono::milliseconds(rows.empty() ? 800 : 155);
      const auto result=cv::search(state,deadline,config,&emergency);
      const auto encoded=result.action.text();
      const double ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
      auto child=state;
      if (!cv::apply_action(child,result.action)) throw std::runtime_error("illegal decision");
      const auto &model=cv::deployment_model();
      const auto base=model.prepare(cv::active_features(state));
      const auto features=cv::active_features(child);
      if (std::bit_cast<std::uint32_t>(model.evaluate(features))!=
          std::bit_cast<std::uint32_t>(model.evaluate_delta(base,features)))
        throw std::runtime_error("actual-model full/delta mismatch");
      for (const auto &root:result.root_actions) {
        auto successor=state;
        if (!cv::apply_action(successor,root.action)) throw std::runtime_error("illegal root action");
        const auto active=cv::active_features(successor);
        if (std::bit_cast<std::uint32_t>(model.evaluate(active))!=
            std::bit_cast<std::uint32_t>(model.evaluate_delta(base,active)))
          throw std::runtime_error("root-action full/delta mismatch");
      }
      rows.push_back({line.substr(0,tab),encoded,trace(result),ms,result.stats.generated_children,
          result.stats.evaluated_children,result.stats.tree_nodes,result.stats.expansions});
    }
    std::cout << std::setprecision(17)
      << "{\"schema\":\"papersoccer.compact-engine-version-probe.v2\",\"mode\":\"" << mode
      << "\",\"payload_sha256\":\"" << cv::deployment_model().payload_sha256()
      << "\",\"all_actions_legal\":true,\"all_root_actions_legal\":true,"
      << "\"actual_model_full_delta_bit_exact\":true,\"all_root_actions_full_delta_bit_exact\":true,\"rows\":[";
    for (std::size_t i=0;i<rows.size();++i) {
      const auto &r=rows[i];if (i) std::cout << ',';
      std::cout << "{\"id\":\"" << r.id << "\",\"action\":\"" << r.action
        << "\",\"milliseconds\":" << r.ms << ",\"generated_successors\":" << r.generated
        << ",\"evaluated_successors\":" << r.evaluated << ",\"nodes\":" << r.nodes
        << ",\"expansions\":" << r.expansions << ",\"fixed_trace\":\"" << r.trace << "\"}";
    }
    std::cout << "]}\n";return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n';return 1; }
}
