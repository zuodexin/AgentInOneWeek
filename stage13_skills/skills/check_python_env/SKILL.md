---
name: check_python_env
description: A skill to check the python environment
license: Apache-2.0
metadata:
  author: example-org
  version: "1.0"
---

# Step-by-step instructions
1. check the python version and the installed packages in the current environment.
2. check if the tools in the project root can be import successfully, if not, return the error message. 

# Examples of inputs and outputs

input: no input needed

output:
- python_version, "the python version in the current environment"
- import_status, "the status of importing tools in the project root, if there is any

# Common edge cases