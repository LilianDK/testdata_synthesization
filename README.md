# Testdata Synthetization


- [Testdata Synthetization](#testdata-synthetization)
  - [Installation](#installation)
  - [Starting the tracing server](#starting-the-tracing-server)


## Preconditions
https://visualstudio.microsoft.com/visual-cpp-build-tools/
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser

## Installation
First create a virtual env.

```bash
python -m venv .venv
```

Then activate the Venv (different on different OS), but windows

```bash
.\.venv\Scripts\activate
```

Then install the python dependencies (don't forget -> need internet!).
```bash
pip install -e .
```

## Starting the tracing server

```bash
python -m phoenix.server.main serve
```

Find the traces on http://localhost:6006