# Sample Size Calculator

A full-stack application for statistical power analysis and sample size calculation across various study designs.

## Features

- Calculate sample size requirements for a desired power level
- Calculate power for a given sample size
- Estimate minimum detectable effects
- Support for multiple study designs:
  - Parallel RCTs with continuous or binary outcomes
  - Cluster RCTs with continuous or binary outcomes
  - Stepped wedge trials
- Interactive web interface built with Streamlit
- REST API powered by FastAPI
- Command-line interface using Typer
- High-performance simulation via Julia integration

## Project Structure

```
sample-size-dashboard/
├── app/
│   └── streamlit_app.py        # Streamlit frontend
├── api/
│   └── main.py                 # FastAPI app
├── core/
│   ├── power.py                # Core calculation functions
│   ├── simulation.py           # Simulation-based methods
│   └── utils.py                # Shared utilities
├── julia_backend/
│   └── stepped_wedge.jl        # High-performance Julia code
├── cli.py                      # Command-line interface
├── tests/
│   ├── test_power.py           # Unit tests for power calculations
│   ├── test_simulation.py      # Unit tests for simulations
│   └── test_julia_interop.py   # Tests for Julia integration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sample-size-dashboard
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install Julia for high-performance simulation:
   - Download and install Julia from https://julialang.org/downloads/
   - Configure PyJulia:
     ```python
     import julia
     julia.install()
     ```

## Usage

### Web Interface

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to http://localhost:8501

### API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs

### Command Line Interface

The CLI provides quick access to calculations:

```bash
# Calculate sample size for a parallel RCT with continuous outcome
python cli.py sample-size --design parallel --outcome continuous --delta 0.5 --std-dev 1.0 --power 0.8

# Calculate power for a cluster RCT with binary outcome
python cli.py power --design cluster --outcome binary --n-clusters 10 --cluster-size 20 --icc 0.05 --p1 0.5 --p2 0.6

# Calculate minimum detectable effect
python cli.py mde --design cluster --outcome binary --n-clusters 15 --cluster-size 25 --icc 0.1 --p1 0.5 --power 0.8
```

### Using Julia for High-Performance Simulation

To test if Julia is correctly configured:

```bash
python cli.py julia
```

## Core Modules

### `core.power`

Contains analytical functions for power and sample size calculation:

- `sample_size_difference_in_means`: Calculate sample size for detecting a difference in means
- `power_difference_in_means`: Calculate power for detecting a difference in means
- `power_binary_cluster_rct`: Calculate power for a cluster RCT with binary outcome
- `sample_size_binary_cluster_rct`: Calculate sample size for a cluster RCT with binary outcome
- `min_detectable_effect_binary_cluster_rct`: Calculate minimum detectable effect for a cluster RCT

### `core.simulation`

Contains simulation-based functions for more complex designs:

- `simulate_parallel_rct`: Simulate a parallel RCT with continuous outcome
- `simulate_cluster_rct`: Simulate a cluster RCT with continuous outcome
- `simulate_stepped_wedge`: Simulate a stepped wedge design
- `simulate_binary_cluster_rct`: Simulate a cluster RCT with binary outcome

### `julia_backend`

Contains high-performance Julia implementations:

- `stepped_wedge.jl`: Fast simulation for stepped wedge designs

## Running Tests

Run all tests:

```bash
pytest
```

Run specific test files:

```bash
pytest tests/test_power.py
pytest tests/test_simulation.py
```

Skip Julia tests if Julia is not installed:

```bash
pytest -k "not julia"
```

## License

[MIT License](LICENSE)
