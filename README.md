_This project has been created as part of the 42 curriculum by vcoevert_

# Call Me Maybe
This project is an introduction into how LLMs work, and how to make them more reliable for Function Calling.

## Instructions
This project uses uv for dependency and environment management, you can install all dependencies using:
```uv sync```
To run the program run the following command:
```uv run python -m src```

A Makefile is provided to make usage and management of this program easier.
It contains the Following targets:
```install```: This installs project dependencies using uv.
```run```: Runs the program.
```debug```: Runs the program in debug mode using pdb.
```clean```: Removes temporary files and cashes.

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
