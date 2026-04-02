
def dpll_satisfiable(formula):
    """
    Determine if a given propositional logic formula is satisfiable using the DPLL algorithm.

    Args:
        formula (list of list of str): A CNF formula represented as a list of clauses,

    """
    
    clauses = [set(clause) for clause in formula]
    symbols = set()
    for clause in clauses:
        for lit in clause:
            if lit.startswith("-"):
                symbols.add(lit[1:])
            else:
                symbols.add(lit)
    symbols = set(sorted(symbols))  # Sort symbols for consistent ordering
    return dpll(clauses, symbols, {})



def dpll(clauses, symbols, model):
    
    # Check if the current model satisfies all clauses
    if all(any((lit in model and model[lit]) or (lit.startswith("-") and lit[1:] in model and not model[lit[1:]]) for lit in clause) for clause in clauses):
        return True
    
    if any(all((lit in model and not model[lit]) or (lit.startswith("-") and lit[1:] in model and model[lit[1:]]) for lit in clause) for clause in clauses):
        return False
    
    symbol, value = find_pure_symbol(clauses, symbols, model)
    if symbol is not None:
        model[symbol] = value
        return dpll(clauses, symbols - {symbol}, model)
    
    symbol, value = find_unit_clause(clauses, model)
    if symbol is not None:
        model[symbol] = value
        return dpll(clauses, symbols - {symbol}, model)

    # Choose a symbol to assign
    unassigned = [s for s in symbols if s not in model]
    if not unassigned:
        return False  # No symbols left to assign, but not all clauses are satisfied

    symbol = unassigned[0]

    # Try assigning True to the symbol
    current_model = model.copy()
    model[symbol] = True
    if dpll(clauses, symbols - {symbol}, model):
        return True

    model = current_model
    model[symbol] = False
    return dpll(clauses, symbols - {symbol}, model)

def find_pure_symbol(clauses, symbols, model):
    for symbol in symbols:
        if symbol in model:
            continue
        positive = any(symbol in clause for clause in clauses)
        negative = any("-" + symbol in clause for clause in clauses)
        if positive and not negative:
            return symbol, True
        if negative and not positive:
            return symbol, False
    return None, None

def find_unit_clause(clauses, model):
    for clause in clauses:
        symbols_in_clause = set(lit if not lit.startswith("-") else lit[1:] for lit in clause)
        if any(sym in model for sym in symbols_in_clause):
            continue  # Skip clauses that already have an assigned symbol
        unassigned = [lit for lit in clause if lit not in model]
        if len(unassigned) == 1:
            value = not unassigned[0].startswith("-")
            symbol = unassigned[0] if not unassigned[0].startswith("-") else unassigned[0][1:]
            return symbol, value
    return None, None

def generate_random_formula(num_clauses, num_symbols):
    import random
    symbols = [f"S{i+1}" for i in range(num_symbols)]
    formula = []
    while len(formula) < num_clauses:
        clause_size = random.randint(1, min(3, num_symbols))  # Random clause size between 1 and 3
        clause = random.sample(symbols, clause_size)
        clause = [lit if random.random() < 0.5 else "-" + lit for lit in clause]  # Randomly negate literals
        if clause not in formula:  # Avoid duplicate clauses
            formula.append(clause)
    return formula

if __name__ == "__main__":
    #formula = generate_random_formula(6, 4)  # Generate a random formula with 6 clauses and 4 symbols
    formula = [['-S2', 'S3'], ['S1', 'S3', 'S2', 'S4'], ['S2', '-S3', 'S4'], ['S4', '-S1', '-S3', '-S2'], ['-S3', '-S2'], ['-S4', 'S3', 'S2', '-S1'], ['S3', 'S4', '-S1', '-S2'], ['S3', '-S4', 'S1', '-S2'], ['S4', '-S2', '-S3'], ['-S4']]
    print("Generated formula:", formula)
    print(dpll_satisfiable(formula))