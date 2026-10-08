from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError, model_validator
from pydantic import validate_call, ConfigDict, PrivateAttr
from typing import Any
import json
import numpy as np

class FunctionCallerError(Exception):
    pass

class FunctionCaller(BaseModel):
    model_name: str = "Qwen/Qwen3-0.6B"
    function_definitions_file: str
    input_prompts_file: str
    output_file: str
    _llm: Small_LLM_Model
    _prompts: list[str] = []
    _func_prompt_start: list[int]
    _func_prompt_end: list[int]
    _stop_token: int

    model_config = ConfigDict(extra='forbid')
    
    @model_validator(mode='after')
    def llm_name_validation(self) -> "FunctionCaller":
        try:
            self._llm = Small_LLM_Model(model_name=self.model_name)
            self._func_prompt_start = self._llm.encode("<|im_start|>system\n"
                                             "Keep response short\n"
                                             "<|im_start|>user\n"
                                             ).tolist()[0]
            self._func_prompt_end = self._llm.encode(" /no_think<|im_end|>"
                                           "\n<|im_start|>assistant\n"
                                           "<think>\n\n</think>\n"
                                           ).tolist()[0]
            self._stop_token = self._llm.encode("<|im_end|>").tolist()[0][0]
            return self
        except OSError:
            "OSError has to be cought because llm_sdk devs were slacking"
            raise ValueError("Invalid model name")

    @model_validator(mode='after')
    def function_file_validation(self) -> "FunctionCaller":
        try:
            with open(self.function_definitions_file, "r") as f:
                "file is openable"
                "TODO: file format checking"
            return self
        except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
            raise ValueError("Could not open function_definitions_file:"
                                  f" {e}")

    @model_validator(mode='after')
    def input_prompt_validation(self) -> "FunctionCaller":
        try:
            with open(self.input_prompts_file, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise TypeError
                for prompt_dict in data:
                    if not isinstance(prompt_dict, dict):
                        raise TypeError
                    wrong_vals = ["Error" for k, v in prompt_dict.items()
                                  if k != "prompt" or not isinstance(v, str)]
                    if len(wrong_vals):
                        raise TypeError
                    self._prompts.append(prompt_dict["prompt"])
            return self
        except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
            raise ValueError("Could not open input_prompts_file:"
                                  f" {e}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in input_prompts_file")
        except TypeError:
            raise ValueError("Wrongly formatted JSON in "
                                  "input_prompts_file")

    @model_validator(mode='after')
    def output_file_validation(self) -> "FunctionCaller":
        try:
            with open(self.output_file, "w") as f:
                "file is openable"
                "TODO: file format checking"
            return self
        except (PermissionError, IsADirectoryError) as e:
            raise ValueError("Could not open output_file:"
                                  f" {e}")

    def call_functions(self) -> bool:
        ret = []
        try:
            for prompt in self._prompts:
                prompt_tokens = self._llm.encode(prompt).tolist()[0]
                user_prompt = (self._func_prompt_start + prompt_tokens +
                               self._func_prompt_end)
                ai_prompt = []
                highest = 0
                while highest != self._stop_token:
                    full_prompt = user_prompt + ai_prompt
                    logits = self._llm.get_logits_from_input_ids(full_prompt)
                    np_logits = np.array(logits)
                    highest = np.argmax(np_logits)
                    ai_prompt.append(highest)
                call = {}
                call["prompt"] = prompt
                call["response"] = f"{self._llm.decode(ai_prompt)}"
                ret.append(call)
                # print(prompt)
                # print(self._llm.decode(ai_prompt))
            with open(self.output_file, "w") as f:
                f.write(json.dumps(ret, indent=4))
            return True
        except FileNotFoundError as e:
            print(e)
            return False


if __name__ == "__main__":
    try:
        caller = FunctionCaller(function_definitions_file="test",
                                input_prompts_file="data/input/function_calling_tests.json",
                                output_file="data/output/function_calling_results.json")
        caller.call_functions()
    except FunctionCallerError as e:
        print(e)
    #     with open("data/input/function_calling_tests.json", "r") as f:
    #         data: list[dict[str, str]] = json.load(f)
    #         print(data)
    #         llm = Small_LLM_Model()
    #         stop_token: int = llm.encode("<|im_end|>").tolist()[0][0]
    #         for entry in data:
    #             print(entry["prompt"], end="\n\n")
    #             tokens: list[int] = llm.encode("<|im_start|>system\n"
    #                                            "Keep response short\n"
    #                                            "<|im_start|>user\n"
    #                                            f"{entry['prompt']}"
    #                                            " /no_think<|im_end|>"
    #                                            "\n<|im_start|>assistant\n"
    #                                            "<think>\n\n</think>\n"
    #                                            ).tolist()[0]
    #             highest = 0
    #             while highest != stop_token:
    #                 logits = llm.get_logits_from_input_ids(tokens)
    #                 np_logits = np.array(logits)
    #                 highest = np.argmax(np_logits)
    #                 print(llm.decode([highest]))
    #                 tokens.append(highest)
    #             print(llm.decode(tokens), "\n\n")
    #
    #
    #
    # except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
    #     print(f"Cannot open file: {e}")
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

