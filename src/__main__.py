from llm_sdk import Small_LLM_Model

if __name__ == "__main__":
    llm = Small_LLM_Model()
    print("here")
    tokens: list[int] = llm.encode('Hey it\'s me it\'s Verity... ASK ME').tolist()[0]
    print("here")
    for _ in range(60):
        logits = llm.get_logits_from_input_ids(tokens)
        logits = logits
        sorted = logits.copy()
        sorted.sort()
        highest = logits.index(sorted[-1])
        print(highest)
        print(llm.decode([highest]))
        tokens.append(highest)
    print(llm.decode(tokens))

