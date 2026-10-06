from llm_sdk import Small_LLM_Model
import json
import numpy as np

if __name__ == "__main__":
    try:
        with open("data/input/function_calling_tests.json", "r") as f:
            data: list[dict[str, str]] = json.load(f)
            print(data)
            llm = Small_LLM_Model()
            for entry in data:
                print(entry["prompt"])
                tokens: list[int] = llm.encode("<|im_start|>user\n"
                                               f"{entry['prompt']}<|im_end|>"
                                               "\n<|im_start|>assistant\n"
                                               ).tolist()[0]
                for _ in range(80):
                    logits = llm.get_logits_from_input_ids(tokens)
                    np_logits = np.array(logits)
                    highest = np.argmax(np_logits)
                    print(highest)
                    print(llm.decode([highest]))
                    tokens.append(highest)
                print(llm.decode(tokens))



    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        print(f"Cannot open file: {e}")
    # llm = Small_LLM_Model()
    # print("here")
    # tokens: list[int] = llm.encode('Hey it\'s me it\'s Verity... ASK ME').tolist()[0]
    # print("here")
    # for _ in range(60):
    #     logits = llm.get_logits_from_input_ids(tokens)
    #     logits = logits
    #     sorted = logits.copy()
    #     sorted.sort()
    #     highest = logits.index(sorted[-1])
    #     print(highest)
    #     print(llm.decode([highest]))
    #     tokens.append(highest)
    # print(llm.decode(tokens))

