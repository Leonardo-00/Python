import random

def line_check(clauses, model):
    return any(all((lit in model and not model[lit]) or (lit.startswith("-") and lit[1:] in model and model[lit[1:]]) for lit in clause) for clause in clauses)

def oracle_check(clauses, model):
    for clause in clauses:
        clause_sat = False
        for lit in clause:
            neg = lit.startswith("-")
            var = lit[1:] if neg else lit
            if var not in model:
                continue
            lit_true = (not neg and model[var]) or (neg and not model[var])
            if lit_true:
                clause_sat = True
                break
        if not clause_sat:
            return False
    return True

def random_formula(symbols, max_clauses=8, max_k=4):
    formula = []
    for _ in range(random.randint(1, max_clauses)):
        k = random.randint(1, min(max_k, len(symbols)))
        vars_ = random.sample(symbols, k)
        clause = [v if random.random() < 0.5 else "-" + v for v in vars_]
        formula.append(clause)
    return formula

def random_partial_model(symbols):
    model = {}
    for s in symbols:
        v = random.choice([None, True, False])
        if v is not None:
            model[s] = v
    return model

def test_check(trials=200000, n_symbols=6, seed=random.randint(1, 1000000)):
    random.seed(seed)
    symbols = [f"S{i+1}" for i in range(n_symbols)]
    for t in range(1, trials + 1):
        formula = random_formula(symbols)
        model = random_partial_model(symbols)
        a = line_check(formula, model)
        b = oracle_check(formula, model)
        if a != b:
            print("Mismatch!")
            print("test:", t)
            print("formula:", formula)
            print("model:", model)
            print("line_check:", a, "oracle_check:", b)
            return False
    print(f"OK: nessun mismatch in {trials} test")
    return True

if __name__ == "__main__":
    test_check()