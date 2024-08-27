# Testdata Synthetization


- [Testdata Synthetization](#testdata-synthetization)
  - [Installation](#installation)
  - [Starting the tracing server](#starting-the-tracing-server)


## Installation

First create a virtual env.

```bash
python -m venv .venv
```

Then activate the Venv (different on different OS), but windows

```bash
.\.venv\Scripts\activate
```

Then install the python dependencies.
```bash
pip install .
```

## Starting the tracing server

```bash
python3 -m phoenix.server.main serve
```

Find the traces on http://localhost:6006