#include <iostream>
#include <string>

#include <tao/pegtl/contrib/parse_tree.hpp>
#include <tao/pegtl/contrib/parse_tree_to_dot.hpp>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "Calc/CalcDialect.h"
#include "Calc/CalcOps.h"
#include "grammar.h"

struct ParserState {
  mlir::MLIRContext context;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  ParserState() : builder(&context) {
    context.getOrLoadDialect<calc::CalcDialect>();
    module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  }
};

// source:
// https://github.com/taocpp/PEGTL/blob/c5eed0d84b5420dc820a7c0a381a87d68f5a1038/src/example/pegtl/parse_tree.cpp#L42
struct rearrange : tao::pegtl::parse_tree::apply<rearrange> {
  // recursively rearrange nodes. the basic principle is:
  //
  // from:          TERM/EXPR
  //                /   |   \          (LHS... may be one or more children,
  //                followed by OP,)
  //             LHS... OP   RHS       (which is one operator, and RHS, which is
  //             a single child)
  //
  // to:               OP
  //                  /  \             (OP now has two children, the original
  //                  TERM/EXPR and RHS)
  //         TERM/EXPR    RHS          (Note that TERM/EXPR has two fewer
  //         children now)
  //             |
  //            LHS...
  //
  // if only one child is left for LHS..., replace the TERM/EXPR with the child
  // directly. otherwise, perform the above transformation, then apply it
  // recursively until LHS... becomes a single child, which then replaces the
  // parent node and the recursion ends.
  template <typename Node, typename... States>
  static void transform(std::unique_ptr<Node> &node, States &&...states) {
    if (node->children.size() == 1) {
      node = std::move(node->children.back());
    } else {
      // node is term or expression
      node->remove_content();
      auto &vec = node->children;

      // pop right hand side from children
      auto rhs = std::move(vec.back());
      vec.pop_back();

      // pop operator node from children
      auto op = std::move(vec.back());
      vec.pop_back();

      // add TERM/EXPR node as first child on the operator node
      op->children.emplace_back(std::move(node));

      // add operator node as second child on the operator node
      op->children.emplace_back(std::move(rhs));

      // replace current node with op node, so it is at the top of the subtree
      node = std::move(op);

      // recursively apply algorithm on LHS
      transform(node->children.front(), states...);
    }
  }
};

template <typename Rule>
using parseTreeSelector = tao::pegtl::parse_tree::selector<
    Rule,
    tao::pegtl::parse_tree::store_content::on<
        // clang-format off
        grammar::integer,
        grammar::identifier,
        grammar::assignment,
        grammar::add_op,
        grammar::mul_op
        // clang-format on
        >,
    rearrange::on<
        // clang-format off
        grammar::term,
        grammar::expression
        // clang-format on
        >>;

int main() {
  std::string inStr;
  std::getline(std::cin, inStr);
  tao::pegtl::memory_input<> in(inStr, "");

  auto root =
      tao::pegtl::parse_tree::parse<grammar::start, parseTreeSelector>(in);

  if (root) {
    tao::pegtl::parse_tree::print_dot(std::cout, *root);
  }

  return 0;
}
