from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError, model_validator
from pydantic import validate_call, ConfigDict, PrivateAttr
from typing import Any
from os import makedirs
from os.path import dirname
import sys
import json
import numpy as np
import re

class FunctionCallerError(Exception):
    pass

class FunctionCaller(BaseModel):
    model_name: str = "Qwen/Qwen3-0.6B"
    function_definitions_file: str
    input_prompts_file: str
    output_file: str
    _llm: Small_LLM_Model
    _prompts: list[str] = []
    _functions: list[dict[str, str | dict[str,str]]] = []
    _function_names: list[str] = []
    _fn_name_tokens: list[list[int]] = []
    _function_params: dict[str, dict[str,str]] = {}
    _name_prompt: list[int]
    _prompt_end: list[int]
    _param_prompt1: list[int]
    _param_prompt2: list[int]
    _stop_token: int

    model_config = ConfigDict(extra='forbid')
    

    def _llm_name_validation(self) -> None:
        try:
            self._llm = Small_LLM_Model(model_name=self.model_name)
            p_name = ("<|im_start|>system\nYou are a function caller, respond"
                      "only with function name from these availabe functions:"
                      f"\n {self._functions}\n<|im_end|><|im_start|>user\n")
            p_end = (" /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n"
                      "\n</think>\n")
            self._name_prompt = self._llm.encode(p_name).tolist()[0]
            self._prompt_end = self._llm.encode(p_end).tolist()[0]
            param_s = ("<|im_start|>system\nYou are a function caller, given"
                       " the function:\n")
            param_e = ("\nrespond only with the right parameters to resolve "
                        "the user's prompt\n<|im_end|><|im_start|>user\n")
            self._param_prompt1 = self._llm.encode(param_s).tolist()[0]
            self._param_prompt2 = self._llm.encode(param_e).tolist()[0]
            self._stop_token = self._llm.encode("<|im_end|>").tolist()[0][0]
            for nm in self._function_names:
                self._fn_name_tokens.append(self._llm.encode(nm).tolist()[0])
        except OSError:
            "OSError has to be cought because llm_sdk devs were slacking"
            raise ValueError("Invalid model name")


    def _initialize_function_definitions(self) -> None:
        try:
            with open(self.function_definitions_file, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise TypeError("Function definitions not part of list")
                for fn in data:
                    if not isinstance(fn, dict):
                        raise TypeError("Function definition not inside "
                                        f"dictionary {fn}")
                    if len(fn) != 4:
                        raise TypeError("Dictionary must contain only name,"
                                        " description, parameters "
                                        f"and returns: {fn}")
                    try:
                        if not isinstance(fn["name"], str):
                            raise TypeError("function name is not a string: "
                                            f"fn")
                        if not isinstance(fn["description"], str):
                            raise TypeError("Function description must be a "
                                            f"string: {fn}")
                        if not isinstance(fn["parameters"], dict):
                            raise TypeError("Function parameters must be a "
                                            f"dictionary: {fn}")
                        for value in fn["parameters"].values():
                            print(value)
                            if not isinstance(value, dict):
                                raise TypeError("Parameters must have a dict "
                                                "containing type: "
                                                f"{fn["parameters"]}")
                            if len(value) != 1:
                                raise TypeError("Parameters dictionary must "
                                                "only contain type: "
                                                f"{fn["parameters"]}")
                            if not isinstance(value["type"], str):
                                raise TypeError("Parameters type must be"
                                                "in string format: "
                                                f"{fn['parameters']}")
                        if not isinstance(fn["returns"], dict):
                            raise TypeError("returns must be a dictionary "
                                            f"containing type: {fn}")
                        if len(fn["returns"]) != 1:
                            TypeError("returns dictionary must only contain "
                                      f"type {fn['returns']}")
                        if not isinstance(fn["returns"]["type"], str):
                            raise TypeError("returns type must be in string "
                                            f"format {fn['returns']}")
                    except KeyError as e:
                        raise TypeError(f"{fn} is missing key {e}")
                    self._functions.append(fn)
                    self._function_names.append(fn["name"])
                    self._function_params[fn["name"]] = fn["parameters"]
        except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
            raise ValueError("Could not open function_definitions_file:"
                                  f" {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in function_definitions_file {e}")
        except TypeError as e:
            raise ValueError("Wrongly formatted JSON in "
                             f"function_definitions_file: {e}")

    def _initialize_prompts(self) -> None:
        try:
            with open(self.input_prompts_file, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise TypeError("Prompts arent part of a list")
                for prompt_dict in data:
                    if not isinstance(prompt_dict, dict):
                        raise TypeError("Prompt not inside dictionary:"
                                        f" {prompt_dict}")
                    wrong_vals = ["Error" for k, v in prompt_dict.items()
                                  if k != "prompt" or not isinstance(v, str)]
                    if len(wrong_vals):
                        raise ValueError("Invalid values inside "
                                         f"input_prompts_file: {wrong_vals}")
                    self._prompts.append(prompt_dict["prompt"])
        except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
            raise ValueError("Could not open input_prompts_file:"
                                  f" {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input_prompts_file: {e}")
        except TypeError as e:
            raise ValueError("Wrongly formatted JSON in "
                             f"input_prompts_file: {e}")

    @staticmethod
    def _output_file_validation(file: str) -> None:
        try:
            makedirs(dirname(file), exist_ok=True)
            with open(file, "w") as f:
                "file is openable in write mode"
        except (PermissionError, IsADirectoryError) as e:
            raise ValueError("Could not open output_file:"
                                  f" {e}")

    @model_validator(mode="after")
    def valiate_model(self)-> "FunctionCaller":
        self._initialize_prompts()
        self._initialize_function_definitions()
        self._output_file_validation(self.output_file)
        self._llm_name_validation()
        return self

    def _llm_pick_from_list(self, prompt: list[int],
                            selection: list[list[int]]) -> list[int]:
        ret: list[int] = []
        while len(selection) > 1:
            for token in selection[1:]:
                if token[0] != selection[0][0]:
                    print(ret)
                    logits = self._llm.get_logits_from_input_ids(prompt)
                    allowed = [l[0] for l in selection]
                    np_logits = np.array(logits)
                    for i in range(len(np_logits)):
                        if i not in allowed:
                            np_logits[i] = -np.inf
                    highest = np.argmax(np_logits)
                    ret.append(highest)
                    prompt.append(highest)
                    selection = [l[1:] for l in selection
                            if len(l) > 1 and l[0] == highest]
                    break
            else:
                ret.append(selection[0][0])
                prompt.append(selection[0][0])
                selection = [l[1:] for l in selection if len(l)]
        if len(selection):
            ret += selection[0]
        return (ret)


    def _get_fn_name(self, prompt: str) -> str:
        prompt_tokens = self._llm.encode(prompt).tolist()[0]
        full_prompt = (self._name_prompt + prompt_tokens +
                       self._prompt_end)
        print(prompt)
        names: list[list[int]] = self._fn_name_tokens.copy()
        pick = self._llm_pick_from_list(full_prompt, names)
        return (self._llm.decode(pick))

    def _get_fn_params(self, prompt: str, name: str) -> dict[str, str]:
        ret = {}
        func_str = f"{name}:{self._function_params[name]}"
        func_tokens = self._llm.encode(func_str).tolist()[0]
        string_tokens = self._llm.encode(prompt).tolist()[0]
        full_prompt = (self._param_prompt1 + func_tokens + 
                       self._param_prompt2 + string_tokens +
                       self._prompt_end)
        s_numbers = re.findall(r'-?\d*\.?\d+', prompt)
        numbers: list[list[int]] = []
        for nb in s_numbers:
            numbers.append(self._llm.encode(nb).tolist()[0])
        print(numbers)
        print(self._llm.decode(string_tokens))
        for k, v in self._function_params[name].items():
            ai_prompt = []
            start = self._llm.encode(f"{k}: ").tolist()[0]
            full_prompt += start
            if v["type"] == "number":
                nbs = numbers.copy()
                ai_prompt = self._llm_pick_from_list(full_prompt, nbs)
                print(self._llm.decode(full_prompt))
                numbers.remove(ai_prompt)
                ai_string = self._llm.decode(ai_prompt)
            else:
                highest = 0
                while highest != self._stop_token:
                    logits = self._llm.get_logits_from_input_ids(full_prompt)
                    np_logits = np.array(logits)
                    for i in range(len(logits)):
                        if i not in string_tokens and i != self._stop_token:
                            np_logits[i] = -np.inf
                    highest = np.argmax(np_logits)
                    if highest != self._stop_token:
                        ai_prompt.append(highest)
                        full_prompt.append(highest)
                        string_tokens.remove(highest)
                        print(self._llm.decode(highest))
                ai_string = self._llm.decode(ai_prompt)
                ai_string = ai_string.strip(" \'").rstrip("\'")
            full_prompt += self._llm.encode(",\n").tolist()[0]
            ret[k] = ai_string
        print(self._llm.decode(full_prompt))
        return ret


    def call_functions(self) -> bool:
        ret = []
        try:
            for prompt in self._prompts:
                call = {
                    "prompt" : prompt,
                    "name": self._get_fn_name(prompt)
                }
                call["parameters"] = self._get_fn_params(prompt, call["name"])
                ret.append(call)
                print(call["parameters"])
            with open(self.output_file, "w") as f:
                _ = f.write(json.dumps(ret, indent=4))
            return True
        except IsADirectoryError as e:
            raise FunctionCallerError(e)


if __name__ == "__main__":
    args = sys.argv[1:]
    func_file="data/input/functions_definition.json"
    input_file="data/input/function_calling_tests.json"
    out_file="data/output/function_calling_results.json"
    try:
        while args:
            match args[0]:
                case "--functions_definition":
                    _ = args.pop(0)
                    func_file = args[0]
                    _ = args.pop(0)
                    if "--functions_definition" in args:
                        raise ValueError("--functions_definition")
                case "--input":
                    _ = args.pop(0)
                    input_file = args[0]
                    _ = args.pop(0)
                    if "--input" in args:
                        raise ValueError("--input")
                case "--output":
                    _ = args.pop(0)
                    out_file = args[0]
                    _ = args.pop(0)
                    if "--output" in args:
                        raise ValueError("--output")
                case _:
                    print(f"skipping invalid parameter: {args[0]}",
                          file=sys.stderr)
                    _ = args.pop(0)
    except ValueError as e:
        print(f"ERROR: Duplicate argument found {e}", file=sys.stderr)
        exit(-1)
    except IndexError:
        print("ERROR: Argument has no parameter", file=sys.stderr)
        exit(-1)

    try:
        caller = FunctionCaller(function_definitions_file=func_file,
                                input_prompts_file=input_file,
                                output_file=out_file)
        caller.call_functions()
    except FunctionCallerError as e:
        print(e)
    except ValidationError as e:
        errorlist = e.errors()
        for errordict in errorlist:
            try:
                errorctx = errordict["ctx"]
                error = errorctx["error"]
            except KeyError:
                error = errordict["msg"]
            print(error)
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

