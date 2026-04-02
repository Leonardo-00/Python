
def backward_chaining(knowledge_base, goal):
    if goal in knowledge_base:
        return True
    for rule in knowledge_base:
        if rule['then'] == goal:
            if all(backward_chaining(knowledge_base, fact) for fact in rule['if']):
                return True
    return False


def main():
    knowledge_base = [
        {'if': ['A', 'B'], 'then': 'M'},
        {'if': ['B'], 'then': 'D'},
        {"if": ["A", "M"], "then": "P"},
        {"if": ["D", "M"], "then": "C"},
        {"if": ["P", "L"], "then": "Q"},
        {"if": ["C", "M"], "then": "L"},
    ]
    goal = 'C'
    result = backward_chaining(knowledge_base, goal)
    print(f"Is '{goal}' inferred? {result}")

if __name__ == "__main__":
    main()