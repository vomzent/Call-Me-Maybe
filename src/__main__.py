try:
    from pydantic import BaseModel, ValidationError, model_validator
    from pydantic import ConfigDict
    from os import makedirs
    from os.path import dirname
    from typing import Any
    import sys
    import json
    import numpy as np
    import re
    import importlib
except ModuleNotFoundError as e:
    print(f"{e} Please make sure you run the project using uv")
    exit(-1)

try:
    llm_sdk: Any = importlib.import_module("llm_sdk")
except ImportError as e:
    print(f"{e} Please make sure the sdk is inside the project folder")
    exit(-1)
except KeyboardInterrupt:
    print("I can't make this part go any quicker")
    exit(-1)


class FunctionCallerError(Exception):
    """Shows errors thrown by the FunctionCaller class

    When the FunctionCaller class runs into an Exception during execution,
    that exception gets cought and the FunctionCaller raises a
    FunctionCallerError with the approriate error message.

    Attributes:
        obj: The object to be cought using try/except (often a message)
    """
    pass


class FunctionCaller(BaseModel):
    """Translates natural language requests into function calls

    The FunctionCaller class takes in a Casual Language Model a function
    definitions file, input prompts file and an output file. Then Translates
    each prompt into a function call that gets written to the output file in
    JSON format.

    Attributes:
        model_name: The name of the Language Model (has to support ChatML)
        function_definitions_file: The file to read function definitons from
        input_prompts_file: The file to read natural language requests from
        output_file: The file to write the function calls to in JSON format
    Raises:
        ValidationError: If during instantiation the configuration is wrong
        FunctionCallerError: if during translation an exception occurs
    """
    model_name: str = "Qwen/Qwen3-0.6B"
    function_definitions_file: str
    input_prompts_file: str
    output_file: str
    _llm: llm_sdk.Small_LLM_Model = None
    _prompts: list[str] = []
    _functions: list[dict[str, str | dict[str, str]]] = []
    _function_names: list[str] = []
    _fn_name_tokens: list[list[np.int32 | np.int64]] = []
    _function_params: dict[str, dict[str, dict[str, str]]] = {}
    _name_prompt: list[np.int32 | np.int64] = []
    _prompt_end: list[np.int32 | np.int64] = []
    _param_prompt1: list[np.int32 | np.int64] = []
    _param_prompt2: list[np.int32 | np.int64] = []
    _stop_tokens: list[np.int32 | np.int64] = []
    model_config = ConfigDict(extra='forbid')

    def _llm_initialization(self) -> None:
        """Instantiates the Small_LLM_Model inside the FunctionCaller class

        This method initializes the Small_LLM_Model and caches the prompts
        the FunctionCaller will use to translate prompts into function calls
        It is not meant to be used outside the class.

        Raises:
            ValueError: if the llm model name isn't found on huggingface
        """
        try:
            self._llm = llm_sdk.Small_LLM_Model(model_name=self.model_name)
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
                       "the user's prompt following this schema:\n"
                       "Name: Value,\n<|im_end|><|im_start|>user\n")
            self._param_prompt1 = self._llm.encode(param_s).tolist()[0]
            self._param_prompt2 = self._llm.encode(param_e).tolist()[0]
            self._stop_tokens = self._llm.encode("<|im_end|>\n").tolist()[0]
            for nm in self._function_names:
                self._fn_name_tokens.append(self._llm.encode(nm).tolist()[0])
        except OSError:
            "OSError has to be cought because llm_sdk devs were slacking"
            raise ValueError("Invalid model name")

    def _initialize_function_definitions(self) -> None:
        """Parses the function definitons file and catches definition errors

        This method opens the function definitions file and tries to
        translate it from JSON to a python object using the json library.
        Then checks that python object for correct formatting.
        It is not meant to be used outside the class.

        Raises:
            ValueError: if an invalid configuration is detected
        """
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
                                            f"{fn["name"]}")
                        if not isinstance(fn["description"], str):
                            raise TypeError("Function description must be a "
                                            f"string: {fn["description"]}")
                        if not isinstance(fn["parameters"], dict):
                            raise TypeError("Function parameters must be a "
                                            f"dictionary: {fn["parameters"]}")
                        for value in fn["parameters"].values():
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
        """Parses the prompts file and catches definition errors

        This method opens the function prompts file and checks if it
        constains correctly formatted prompts, which are then added to a
        big list of prompts.
        It is not meant to be used outside the class.

        Raises:
            ValueError: If an invalid configuration is detected
        """
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
        """Opens the output file to check if it can write into it

        This method tries to make the paths to the output file then tries
        to open the output file, if an error occurs the filename is deemed a
        configuration error an a ValueError is raised.
        It is not Meant to be used outside the class.

        Raises:
            ValueError: of the file couldn't be written into.
        """
        try:
            makedirs(dirname(file), exist_ok=True)
            with open(file, "w") as f:
                "file is openable in write mode"
                f.name
        except (PermissionError, IsADirectoryError) as e:
            raise ValueError(f"Could not open output_file: {e}")

    @model_validator(mode="after")
    def valiate_model(self) -> "FunctionCaller":
        """Validates the configuration of the FunctionCaller

        This method uses pydantic for base type checking. Then the aditional
        above initialization and validation methods. When an error from those
        methods is raised pydantic catches it and wraps in inside a
        ValidationError.
        This method is meant to only be called by pydantic on initialization

        Returns:
            self: A FunctionCaller object

        Raises:
            ValidationError: if configuration is invalid or an error occurs
        """
        self._initialize_prompts()
        self._initialize_function_definitions()
        self._output_file_validation(self.output_file)
        self._llm_initialization()
        return self

    def _llm_pick_from_list(self, prompt: list[np.int32 | np.int64],
                            selection: list[list[np.int32 | np.int64]]
                            ) -> list[np.int32 | np.int64]:
        """Picks one of the token lists in selection using the LLM

        This method takes in the full token history (prompt) and uses the LLM
        together with constraint decoding and preselection to pick one of the
        token lists inside selection. For example given the tokens for
        [[fn,_add,_numbers],[fn,_add,_strings]] the method will preselect the
        tokens fn and _add then use the llm to pick an option where nothing
        can be preselected.
        This method is meant to only be used inside the class.

        Args:
            prompt: The full token history (prompt) as context for the LLM
            selection: A list of token lists the LLM will choose from

        Returns:
            List[int]: A list of the tokens selected
        """
        ret: list[np.int32 | np.int64] = []
        while len(selection) > 1:
            for token in selection[1:]:
                if token[0] != selection[0][0]:
                    logits = self._llm.get_logits_from_input_ids(prompt)
                    allowed = [li[0] for li in selection]
                    np_logits = np.array(logits)
                    for i in range(len(np_logits)):
                        if i not in allowed:
                            np_logits[i] = -np.inf
                    highest = np.argmax(np_logits)
                    ret.append(highest)
                    prompt.append(highest)
                    selection = [li[1:] for li in selection
                                 if len(li) > 1 and li[0] == highest]
                    print(f"LLM has picked: {self._llm.decode(highest)}")
                    break
            else:
                ret.append(selection[0][0])
                prompt.append(selection[0][0])
                print(f"Preselected token: {self._llm.decode(ret[-1])}")
                selection = [li[1:] for li in selection if len(li)]
        if len(selection):
            ret += selection[0]
        print(f"Pick from list picked: {self._llm.decode(ret)}")
        return (ret)

    def _llm_constrainted_decode(self, prompt: list[np.int32 | np.int64],
                                 constaint: list[np.int32 | np.int64]
                                 ) -> list[np.int32 | np.int64]:
        """Picks tokens from constraint using the LLM

        This method takes in the full token history (prompt) and a list of
        constraint tokens. Then makes the LLM pick tokens only part
        of the constraint tokens list. returning when there are no tokens left
        or when the LLM picks a stop token.
        This method is meant to only be used inside the class.

        Args:
            prompt: The full token history (prompt) as context for the LLM
            constraint: A list of tokens the AI is constrainted to choose

        Returns:
            List[int]: A list of tokens selected by the LLM
        """
        ret: list[np.int32 | np.int64] = []
        highest = np.array([0])[0]
        while highest not in self._stop_tokens:
            logits = self._llm.get_logits_from_input_ids(prompt)
            np_logits = np.array(logits)
            for i in range(len(logits)):
                if i not in constaint and i not in self._stop_tokens:
                    np_logits[i] = -np.inf
            highest = np.argmax(np_logits)
            print("Constrainted LLM has generated: "
                  f"{self._llm.decode(highest)}")
            if highest not in self._stop_tokens:
                ret.append(highest)
                prompt.append(highest)
                constaint.remove(highest)
        for _ in ret:
            prompt.pop()
        ret_s = self._llm.decode(ret)
        ret_s = ret_s.strip(" \'\"").rstrip("\'\"")
        ret = self._llm.encode(ret_s).tolist()[0]
        print(f"Constrainted decode returns: {self._llm.decode(ret)}")
        return ret

    def _llm_decode(self, prompt: list[np.int32 | np.int64]
                    ) -> list[np.int32 | np.int64]:
        """Picks tokens using the LLM

        This method takes in the full token history (prompt) and uses the LLM
        to pick the next most probable tokens until the LLM picks a stop
        token. Returning a list of tokens picked.
        This method is only meant to be used inside the class.

        Args:
            prompt: The full token history (prompt) as context for the LLM

        Returns:
            List[int]: A list of tokens selected by the LLM
        """
        ret: list[np.int32 | np.int64] = []
        highest = np.array([0])[0]
        while highest not in self._stop_tokens:
            logits = self._llm.get_logits_from_input_ids(prompt)
            np_logits = np.array(logits)
            highest = np.argmax(np_logits)
            if highest not in self._stop_tokens:
                ret.append(highest)
                prompt.append(highest)
                print(f"LLM has generated: {self._llm.decode(highest)}")
        for _ in ret:
            prompt.pop()
        ret_s = self._llm.decode(ret)
        if ret_s.find(",") != -1:
            ret_s = ret_s[:ret_s.find(",")]
        ret_s = ret_s.strip(" \'\"").rstrip("\'\"")
        ret = self._llm.encode(ret_s).tolist()[0]
        return ret

    def _get_fn_name(self, prompt: str) -> str:
        """Picks a function name given a user's prequest

        This method takes a natural language request as string then picks the
        most likely function to resolve the user's prompt.
        This method is only meant to be used inside the class.

        Args:
            prompt: The natural language request to pick a function name from

        Returns:
            str: The function name picked as a string
        """
        prompt_tokens = self._llm.encode(prompt).tolist()[0]
        full_prompt = (self._name_prompt + prompt_tokens +
                       self._prompt_end)
        names: list[list[np.int32 | np.int64]] = self._fn_name_tokens.copy()
        pick = self._llm_pick_from_list(full_prompt, names)
        ret: str = self._llm.decode(pick)
        return (ret)

    def _get_fn_params(self, prompt: str, name: str) -> dict[str, str]:
        """Picks function parameters given a user's request and function name

        This method takes a natural language request and the name of the
        function. Then picks parameters using constraint decoding and the LLM.
        This method is only meant to be used inside the class

        Args:
            prompt: The natural language request as context for the LLM
            name: The name of the function chosen

        Returns:
            dict[str, str]: Dictionary containing the parameter and value pair
        """
        ret = {}
        func_str = f"{name}:{self._function_params[name]}"
        func_tokens = self._llm.encode(func_str).tolist()[0]
        prompt_tokens = self._llm.encode(prompt).tolist()[0]
        full_prompt = (self._param_prompt1 + func_tokens +
                       self._param_prompt2 + prompt_tokens +
                       self._prompt_end)
        s_numbers = re.findall(r'-?\d*\.?\d+', prompt)
        numbers: list[list[np.int32 | np.int64]] = []
        for nb in s_numbers:
            numbers.append(self._llm.encode(nb).tolist()[0])
        print(self._llm.decode(prompt_tokens))
        for k, v in self._function_params[name].items():
            ai_prompt = []
            start = self._llm.encode(f"{k}: ").tolist()[0]
            full_prompt += start
            if v["type"] == "number":
                nbs = numbers.copy()
                ai_prompt = self._llm_pick_from_list(full_prompt, nbs)
                numbers.remove(ai_prompt)
            elif k == "regex":
                full_prompt += self._llm.encode("r'").tolist()[0]
                ai_prompt = self._llm_decode(full_prompt)
            else:
                ai_prompt = self._llm_constrainted_decode(full_prompt,
                                                          prompt_tokens)
            full_prompt += ai_prompt
            ai_string = self._llm.decode(ai_prompt)
            full_prompt += self._llm.encode(",\n").tolist()[0]
            ret[k] = ai_string
        print(self._llm.decode(full_prompt))
        return ret

    def call_functions(self) -> bool:
        """Generates precise function calls from natural language requests

        This method takes in the configuration given on initialization and
        for each natural language request translates it into a function call
        using the function definitons. Then writes tose translations to a JSON
        file.

        Returns:
            bool: Always True unless an exception is raised

        Raises:
            FunctionCallerError: When an Exception occurs during translation
        """
        ret = []
        try:
            for prompt in self._prompts:
                print(f"\n\n\nPrompt: {prompt}")
                call: dict[str, Any] = {
                    "prompt": prompt,
                    "name": self._get_fn_name(prompt)
                }
                call["parameters"] = self._get_fn_params(prompt,
                                                         call["name"])
                ret.append(call)
                print(call["parameters"])
            with open(self.output_file, "w") as f:
                _ = f.write(json.dumps(ret, indent=4))
            return True
        except Exception as e:
            raise FunctionCallerError(e)


if __name__ == "__main__":
    args = sys.argv[1:]
    func_file = "data/input/functions_definition.json"
    input_file = "data/input/function_calling_tests.json"
    out_file = "data/output/function_calling_results.json"
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
        print("All loaded up!")
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
    except KeyboardInterrupt:
        print("Stopping i guess >:(")
