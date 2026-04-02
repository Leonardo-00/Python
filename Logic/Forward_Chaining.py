
def forward_chaining(knowledge_base, facts):
    inferred = set()
    while True:
        new_inferences = set()
        for rule in knowledge_base:
            if rule["then"] not in inferred and all(fact in facts for fact in rule['if']):
                new_inferences.add(rule['then'])
        if not new_inferences - inferred:
            break
        inferred.update(new_inferences)
        facts.update(new_inferences)
    return inferred


def main():
    knowledge_base = [
        {'if': ['A', 'B'], 'then': 'M'},
        {'if': ['B'], 'then': 'D'},
        {"if": ["A", "M"], "then": "P"},
        {"if": ["D", "M"], "then": "C"},
        {"if": ["P", "L"], "then": "Q"},
        {"if": ["C", "M"], "then": "L"},
    ]
    facts = {'A', 'B'}
    result = forward_chaining(knowledge_base, facts)
    print("Inferred facts:", result)
    
    
if __name__ == "__main__":
    main()