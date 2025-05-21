# DesignPower: Power and Sample Size Calculator

A comprehensive application for statistical power analysis and sample size calculation across various study designs, featuring both analytical and simulation-based methods.

## Features

### Calculation Types
- **Sample Size Calculation**: Determine required sample sizes for a desired power level
- **Power Calculation**: Calculate statistical power for a given sample size
- **Minimum Detectable Effect**: Estimate the smallest effect size detectable with a given sample size and power

### Study Designs
- **Parallel RCTs** with multiple outcome types:
  - Continuous outcomes (with equal or unequal variance options)
  - Binary outcomes (with various statistical test options)
  - Survival outcomes
- **Single Arm Trials** with multiple outcome types
  - Continuous outcomes
  - Binary outcomes
    - Standard design
    - A'Hern's design
    - Simon's two-stage design
  - Survival outcomes
- **Cluster RCTs** with multiple outcome types:
  - Continuous outcomes (accounting for intracluster correlation)
  - Binary outcomes (accounting for intracluster correlation)

### Hypothesis Types
- **Superiority**: Testing if a new treatment is better than a control
- **Non-Inferiority**: Testing if a new treatment is not worse than a control by a pre-specified margin

### Calculation Methods
- **Analytical**: Based on established statistical formulas
- **Simulation**: Monte Carlo simulations for more complex designs and robust estimates

## Advanced Features

### Binary Outcome Advanced Options

#### Statistical Test Types
- **Normal Approximation**: Standard chi-square test approach (z-test)
- **Fisher's Exact Test**: More conservative exact test, ideal for small sample sizes
- **Likelihood Ratio Test**: Often more powerful than chi-square tests

#### Continuity Correction
- Optional continuity correction for improved accuracy in discrete data

### Single Arm Binary Outcome Designs

#### A'Hern's Design
- Exact binomial calculations for smaller sample sizes
- More precise than normal approximation methods
- Single-stage design with clear decision rules

#### Simon's Two-Stage Design
- Allows early stopping for futility after the first stage
- Reduces expected sample size when treatment is ineffective
- Multiple optimality criteria:
  - **Optimal**: Minimizes expected sample size under null hypothesis
  - **Minimax**: Minimizes maximum sample size
- Provides stage-specific decision thresholds
- Highly optimized calculation algorithm based on industry-standard methods

### Simulation Features
- **Customizable Simulations**: Set the number of simulations and random seed
- **Reproducibility**: Fixed random seeds ensure reproducible results
- **Parameter Optimization**: Automatic optimization for sample size and MDE calculations

### Unequal Variance Support
- For continuous outcomes, option to specify different standard deviations for treatment and control groups

## Project Structure

```
DesignPower/
├── app/
│   ├── designpower_app.py     # Streamlit main application
│   └── components/            # Modular UI components
│       ├── parallel_rct.py    # Parallel RCT components
│       └── single_arm.py      # Single arm trial components
├── core/
│   ├── compatibility.py       # Backward compatibility
│   ├── power.py               # Main interface for calculations
│   └── designs/               # Design-specific implementations
│       ├── parallel/          # Parallel design modules
│       │   ├── analytical_continuous.py  # Analytical methods for continuous outcomes
│       │   ├── simulation_continuous.py  # Simulation methods for continuous outcomes
│       │   ├── analytical_binary.py      # Analytical methods for binary outcomes
│       │   ├── simulation_binary.py      # Simulation methods for binary outcomes
│       │   ├── analytical_survival.py    # Analytical methods for survival outcomes
│       │   └── simulation_survival.py    # Simulation methods for survival outcomes
│       └── single/            # Single arm design modules
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DesignPower.git
   cd DesignPower
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App

```bash
cd DesignPower
streamlit run app/designpower_app.py
```

This will start the web application locally and open a browser window with the interface.

### Using the Application

1. **Select Study Design and Outcome Type**
   - Choose between Parallel RCT or Single Arm Trial
   - Select the outcome type (Continuous, Binary, or Survival)

2. **Choose Hypothesis Type**
   - Select between Superiority or Non-Inferiority hypothesis

3. **Select Calculation Type**
   - Sample Size: Calculate required sample size for a given power
   - Power: Calculate power for a given sample size
   - Minimum Detectable Effect: Determine the smallest effect size detectable

4. **Input Parameters**
   - Enter basic parameters specific to your chosen design
   - Configure advanced options as needed (test type, simulation parameters, etc.)

5. **Calculate Results**
   - Click the Calculate button to perform the analysis
   - View results including visualizations

### Advanced Options

#### Simulation vs. Analytical Methods

For more complex designs or to verify analytical results, the simulation method provides:  
- More robust estimates through Monte Carlo simulation
- Customizable number of simulations (higher = more precision)
- Ability to set a random seed for reproducibility

#### Binary Outcome Test Types

Different test types affect the calculated sample size and power:
- **Normal Approximation**: Standard approach, efficient for larger samples
- **Fisher's Exact Test**: More conservative, better for smaller samples
- **Likelihood Ratio Test**: Often more powerful than chi-square tests

Applying continuity correction improves accuracy but generally increases required sample sizes.

## Modular Structure

The application follows a modular design pattern:
- **UI Components**: Separated by study design and outcome type
- **Calculation Modules**: Organized by analytical vs. simulation methods
- **Core Functions**: Shared utilities and interfaces

## Contributing

Contributions to DesignPower are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
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
