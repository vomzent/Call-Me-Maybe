_This project has been created as part of the 42 curriculum by vcoevert_

# Call Me Maybe
This project is an introduction into how LLMs work, and how to make them more reliable for Function Calling.

## Instructions
This project uses uv for dependency and environment management, you can install all dependencies using:
```uv sync```<br>
To run the program run the following command:
```uv run python -m src```

A Makefile is provided to make usage and management of this program easier.<br>
It contains the Following targets:<br>
```install```: This installs project dependencies using uv.<br>
```run```: Runs the program.<br>
```debug```: Runs the program in debug mode using pdb.<br>
```clean```: Removes temporary files and cashes.<br>
```lint```: Runs flake8 and mypy with mandatory flags<br>
```lint-strict```: runs flake8 and mypy with optional flags<br>

## Resources
My peers!!! Including but not limited to:
- cpfister
- ilclaass
- ivan-der
- maavan-d
- odschreu

prompt formatting from the qwen documentation:<br>
https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#chatml

JSON usage from the python docs:<br>
https://docs.python.org/3/library/json.html

regex cheat sheet:<br>
https://www.rexegg.com/regex-quickstart.php

AI was used every time i ran the project.<br>
The project uses the Qwen-3-0.6B model for turning human language prompts into function calls.<br>
Outside of the llm_sdk no external LLMs were used.

## Algorithm explanation
Constraint decoding in this project is done by first getting a list of allowed tokens.<br>
For the _llm_pick_from_list method the allowed tokens are constrainted to be the current index of every token list in selection.<br>
for the _llm_constrainted_decode method the allowed tokens are constrainted to be tokens inside the user's prompt<br>
Because we do not generate JSON with the LLM we never have to identify which tokens would maintain valid JSON structure and complience with the expected schema.<br>
The LLM then produces logits for all tokens. We set all tokens not part of allowed to negative infinity.<br>
We then use numpy to get the index of the highest logit, which will be one of the tokens in allowed tokens.

## Design decisions
The computers at my campus are not optimized for LLMs, nor is my personal computer at home.<br>
The campus cluster computers run Ubuntu with an i5-7500 cpu and an rx 470 gpu both of which can't use hardware acceleration.<br>
Considering the project has to run within 5 minutes. The only thing i could do was use the LLM as little as possible.<br>
JSON complience is thus enforced by python code.
The LLM generates and picks function names and parameter arguments without ever seeing the end result created by the code.<br>
To limit LLM calls even more _llm_pick_from_list was created.
Which only calls the LLM when a selection needs to be made and code cannot preselect a guaranteed value.<br>

## Performance analysis
My solution runs the subject provided test prompts in roughly 1m25s on campus computers.
Including the loading of the llm_sdk library and the initialisation of the LLM.<br>
Before introducing _llm_pick_from_list this speed was drastically slower due to the lack of hardware acceleration on the machines tested.
Taking roughly 4 and a half minutes just to generate every function name.<br>
The solution is also very reliable. Ensuring 100% valid JSON by not letting the LLM decide and always programmatically creating JSON via the python json module.<br>
Accuracy varies. Prompts for function calls that require numbers will be very reliable, as my solution first parses every number inside the user prompt then ensures the LLM picks one of the numbers if the function parameter requires a number.<br>
In contrast to regex, who's constraint decoding had to be turned off because regex functions require charachters and tokens not part of the prompt.<br>

## Challenges faced
The first challenge encountered was the lack of documentation.<br>
The subject requires we use the provided llm_sdk.
However this llm_sdk comes with no explanation or documentation regarding use or installation.
With many peers just deciding not to install it entirely and to instead hardcode it into their project.<br>
The second challenge was the slow loading times of the llm_sdk module, initialisation of the LLM and the slowness of every LLM call.<br>
It made it getting used to the module extremely tedious and difficult.
As every trial run with it would take half a minute to start up.<br>
The third big challenge i faced was wat to do with regex.
It couldn't really be solved using the same constraint decoding methods i made and before.
And i also didn't want to prompt engineer a solution to it considering that's not what the subject is trying to teach.<br>
I ended up deciding that the current solution of constraint decoding some parts and fully relying on the llm for the regex parameter shows enough that i know what i'm doing.<br>
The last major hurdle i faced were flake8 and mypy.
Due to the bad configuration of the llm_sdk module and the nature of this project certain mypy errors were abundant were they shouldn't be.<br>
For example: mypy complained my lists of ints did not support appending numpy integers. and that my lists should explicitly be of type list[np.int32 | np.int64].
This makes no sense ofcourse, because my code runs just fine with tokens represented sometimes by ints sometimes by numpy ints.<br>
But mypy disagreed. Also on Small_LLM_Model.
It kept complaining the llm_sdk had no such thing. Even when it clearly does.<br>
I eventually had to resort to importing llm_sdk via importlib and telling mypy to treat it as Any.
Which finally shut mypy up.

## Testing strategy
Testing was done by running the project manually after every code change.
Inspecting the data/output/function_calling_results.json file. And testing both valid and invalid inputs.
Careful consideration and significant time was invested into making sure the program works reliably and behaves as expected.<br>

## Example usage
To run the program first all dependencies must get installed, this can be done either using `uv sync` or typing `make run`.<br>
Then the program can be run with default arguments using `make run` or `uv run python -m src`.<br>
The program supports custom paths for the functions definiton file, the input prompts file and the output file.
These can be specified using the following arguments: `--functions_definition` `--input` `--output`
