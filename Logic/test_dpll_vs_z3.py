import argparse
import random
from typing import List

from z3 import Bool, Not, Or, Solver, sat

from DPLL import dpll, dpll_satisfiable

Formula = List[List[str]]


def z3_check_with_model(formula: Formula) -> tuple[bool, dict]:
    """Check formula with Z3 and return (satisfiable, model)."""
    solver = Solver()
    symbols = {}

    def symbol(name: str):
        if name not in symbols:
            symbols[name] = Bool(name)
        return symbols[name]

    for clause in formula:
        z3_clause = []
        for lit in clause:
            if lit.startswith("-"):
                z3_clause.append(Not(symbol(lit[1:])))
            else:
                z3_clause.append(symbol(lit))
        solver.add(Or(*z3_clause))

    is_sat = solver.check() == sat
    model = {}
    if is_sat:
        z3_model = solver.model()
        for sym_name, z3_var in symbols.items():
            model[sym_name] = z3_model[z3_var]
    return is_sat, model


def dpll_check_with_model(formula: Formula) -> tuple[bool, dict]:
    """Check formula with DPLL and return (satisfiable, model)."""
    clauses = [set(clause) for clause in formula]
    symbols = set()
    for clause in clauses:
        for lit in clause:
            if lit.startswith("-"):
                symbols.add(lit[1:])
            else:
                symbols.add(lit)
    model = {}
    result = dpll(clauses, symbols, model)
    return result, model


def random_formula(
    rng: random.Random,
    max_symbols: int,
    max_clauses: int,
    max_clause_size: int,
) -> Formula:
    symbol_count = rng.randint(1, max_symbols)
    symbols = [f"S{i + 1}" for i in range(symbol_count)]
    clause_count = rng.randint(1, max_clauses)

    formula: Formula = []
    for _ in range(clause_count):
        clause_size = rng.randint(1, min(max_clause_size, symbol_count))
        clause_symbols = rng.sample(symbols, clause_size)
        clause = [sym if rng.random() < 0.5 else f"-{sym}" for sym in clause_symbols]
        formula.append(clause)

    return formula


def run_differential_tests(
    test_count: int,
    seed: int,
    max_symbols: int,
    max_clauses: int,
    max_clause_size: int,
) -> bool:
    rng = random.Random(seed)

    for index in range(1, test_count + 1):
        formula = random_formula(rng, max_symbols, max_clauses, max_clause_size)
        dpll_result, dpll_model = dpll_check_with_model(formula)
        z3_result, z3_model = z3_check_with_model(formula)

        if dpll_result != z3_result:
            print(f"\n{'='*70}")
            print(f"Mismatch found on test #{index}")
            print(f"{'='*70}")
            print(f"Formula:")
            for i, clause in enumerate(formula, 1):
                print(f"  {i}. {clause}")
            print(f"\nDPLL result: {dpll_result}")
            print(f"DPLL model:  {dpll_model}")
            print(f"\nZ3 result:   {z3_result}")
            print(f"Z3 model:    {z3_model}")
            print(f"{'='*70}\n")
            return False

    print(
        f"No mismatches found across {test_count} random formulas "
        f"(seed={seed}, max_symbols={max_symbols}, max_clauses={max_clauses}, max_clause_size={max_clause_size})."
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Differential test: compare custom DPLL solver vs Z3 SAT solver."
    )
    parser.add_argument("--tests", type=int, default=1000, help="Number of random CNFs to test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--max-symbols", type=int, default=6, help="Maximum distinct symbols.")
    parser.add_argument("--max-clauses", type=int, default=12, help="Maximum number of clauses.")
    parser.add_argument("--max-clause-size", type=int, default=4, help="Maximum literals per clause.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.tests <= 0:
        raise ValueError("--tests must be positive")
    if args.max_symbols <= 0:
        raise ValueError("--max-symbols must be positive")
    if args.max_clauses <= 0:
        raise ValueError("--max-clauses must be positive")
    if args.max_clause_size <= 0:
        raise ValueError("--max-clause-size must be positive")

    ok = run_differential_tests(
        test_count=args.tests,
        seed=args.seed,
        max_symbols=args.max_symbols,
        max_clauses=args.max_clauses,
        max_clause_size=args.max_clause_size,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
