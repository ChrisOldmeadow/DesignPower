"""
Command Line Interface for sample size calculator.

This module provides a CLI for sample size calculation, power analysis,
and simulation-based estimation using the typer package.
"""
import typer
from typing import Optional
from enum import Enum
from rich.console import Console
from rich.table import Table
import json

from core.power import (
    sample_size_difference_in_means,
    power_difference_in_means,
    power_binary_cluster_rct,
    sample_size_binary_cluster_rct,
    min_detectable_effect_binary_cluster_rct
)
from core.designs.single_arm.continuous import (
    one_sample_t_test_sample_size,
    one_sample_t_test_power
)
from core.designs.single_arm.binary import (
    one_sample_proportion_test_sample_size,
    one_sample_proportion_test_power,
    ahern_sample_size,
    ahern_power,
    simons_two_stage_design,
    simons_power
)
from core.designs.single_arm.survival import (
    one_sample_survival_test_sample_size,
    one_sample_survival_test_power
)
from core.designs.parallel.analytical_survival import (
    sample_size_survival,
    power_survival,
    min_detectable_effect_survival,
    sample_size_survival_non_inferiority,
    power_survival_non_inferiority
)
from core.simulation import (
    simulate_parallel_rct,
    simulate_cluster_rct,
    simulate_stepped_wedge,
    simulate_binary_cluster_rct
)
from core.utils import generate_plain_language_summary

app = typer.Typer(help="Sample Size and Power Calculation Tool")
console = Console()


class DesignType(str, Enum):
    """Enum for supported study designs."""
    PARALLEL = "parallel"
    CLUSTER = "cluster"
    STEPPED_WEDGE = "stepped-wedge"
    SINGLE_ARM = "single-arm"


class OutcomeType(str, Enum):
    """Enum for supported outcome types."""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    SURVIVAL = "survival"


class CalculationType(str, Enum):
    """Enum for calculation types."""
    SAMPLE_SIZE = "sample-size"
    POWER = "power"
    MDE = "mde"  # Minimum Detectable Effect


def display_result(result, method_name):
    """Display calculation result in a formatted table."""
    # Display main result
    main_table = Table(title="Calculation Result")
    
    # Add columns dynamically based on result keys
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", style="green")
    
    for key, value in result.items():
        if key != "parameters" and not isinstance(value, dict):
            # Format float values
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            main_table.add_row(key.replace("_", " ").title(), formatted_value)
    
    console.print(main_table)
    
    # Display parameters
    param_table = Table(title="Input Parameters")
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Value", style="green")
    
    for key, value in result.get("parameters", {}).items():
        # Format float values
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        param_table.add_row(key.replace("_", " ").title(), formatted_value)
    
    console.print(param_table)
    
    # Display plain language summary
    summary = generate_plain_language_summary(method_name, result)
    console.print("\n[bold]Summary:[/bold]")
    console.print(summary)


@app.command()
def sample_size(
    design: DesignType = typer.Option(DesignType.PARALLEL, help="Study design"),
    outcome: OutcomeType = typer.Option(OutcomeType.CONTINUOUS, help="Outcome type"),
    delta: Optional[float] = typer.Option(None, help="Minimum detectable effect (difference in means)"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation (for continuous outcomes)"),
    p1: Optional[float] = typer.Option(None, help="Proportion in control group (for binary outcomes)"),
    p2: Optional[float] = typer.Option(None, help="Proportion in intervention group (for binary outcomes)"),
    power: float = typer.Option(0.8, help="Desired statistical power (1 - beta)"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    allocation_ratio: float = typer.Option(1.0, help="Ratio of sample sizes (n2/n1) for parallel design"),
    cluster_size: Optional[int] = typer.Option(None, help="Average number of individuals per cluster (for cluster designs)"),
    icc: Optional[float] = typer.Option(None, help="Intracluster correlation coefficient (for cluster designs)"),
    steps: Optional[int] = typer.Option(None, help="Number of time steps (for stepped wedge design)"),
    individuals_per_cluster: Optional[int] = typer.Option(None, help="Number of individuals per cluster per step (for stepped wedge)"),
    treatment_effect: Optional[float] = typer.Option(None, help="Effect size of intervention (for stepped wedge)"),
    nsim: int = typer.Option(1000, help="Number of simulations (for simulation-based estimation)"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    generate_script: bool = typer.Option(False, "--script", help="Generate reproducible Python script instead of just showing results"),
    script_output: Optional[str] = typer.Option(None, "--script-file", help="Save generated script to file (use with --script)")
):
    """
    Calculate required sample size for a given study design.
    """
    try:
        # If script generation is requested, use the new comprehensive script generator
        if generate_script:
            # Build parameters for script generation
            params = {
                'calculation_type': 'Sample Size',
                'method': 'analytical',  # Default to analytical for CLI
                'alpha': alpha,
                'power': power,
                'allocation_ratio': allocation_ratio,
                'nsim': nsim
            }
            
            # Handle different design/outcome combinations
            if design == DesignType.PARALLEL:
                if outcome == OutcomeType.CONTINUOUS:
                    if delta is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For continuous outcomes, --delta and --std-dev are required.")
                        raise typer.Exit(1)
                    params.update({'mean1': 0.0, 'mean2': delta, 'std_dev': std_dev})
                    from app.components.parallel_rct import generate_cli_code_parallel_continuous
                    script = generate_cli_code_parallel_continuous(params)
                    
                elif outcome == OutcomeType.BINARY:
                    if p1 is None or p2 is None:
                        console.print("[bold red]Error:[/bold red] For binary outcomes, --p1 and --p2 are required.")
                        raise typer.Exit(1)
                    params.update({'p1': p1, 'p2': p2})
                    from app.components.parallel_rct import generate_cli_code_parallel_binary
                    script = generate_cli_code_parallel_binary(params)
                    
            elif design == DesignType.CLUSTER:
                if cluster_size is None or icc is None:
                    console.print("[bold red]Error:[/bold red] For cluster designs, --cluster-size and --icc are required.")
                    raise typer.Exit(1)
                    
                if outcome == OutcomeType.CONTINUOUS:
                    if delta is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For continuous outcomes, --delta and --std-dev are required.")
                        raise typer.Exit(1)
                    params.update({
                        'cluster_size': cluster_size, 'icc': icc,
                        'mean1': 0.0, 'mean2': delta, 'std_dev': std_dev
                    })
                    from app.components.cluster_rct import generate_cli_code_cluster_continuous
                    script = generate_cli_code_cluster_continuous(params)
                    
                elif outcome == OutcomeType.BINARY:
                    if p1 is None or p2 is None:
                        console.print("[bold red]Error:[/bold red] For binary outcomes, --p1 and --p2 are required.")
                        raise typer.Exit(1)
                    params.update({
                        'cluster_size': cluster_size, 'icc': icc,
                        'p1': p1, 'p2': p2
                    })
                    from app.components.cluster_rct import generate_cli_code_cluster_binary
                    script = generate_cli_code_cluster_binary(params)
            else:
                console.print(f"[bold red]Error:[/bold red] Script generation not yet supported for {design.value} design.")
                raise typer.Exit(1)
            
            # Output the script
            if script_output:
                with open(script_output, 'w') as f:
                    f.write(script)
                console.print(f"[bold green]✓[/bold green] Reproducible script saved to: {script_output}")
                console.print(f"[bold blue]Usage:[/bold blue] python {script_output}")
            else:
                console.print(script)
            return
        
        # Original calculation logic for basic results display
        result = None
        method_name = None
        
        # Parallel design with continuous outcome
        if design == DesignType.PARALLEL and outcome == OutcomeType.CONTINUOUS:
            if delta is None or std_dev is None:
                console.print("[bold red]Error:[/bold red] For continuous outcomes, --delta and --std-dev are required.")
                raise typer.Exit(1)
            
            result = sample_size_difference_in_means(
                delta=delta,
                std_dev=std_dev,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio
            )
            method_name = "sample_size_difference_in_means"
        
        # Cluster design with binary outcome
        elif design == DesignType.CLUSTER and outcome == OutcomeType.BINARY:
            if p1 is None or p2 is None or cluster_size is None or icc is None:
                console.print("[bold red]Error:[/bold red] For cluster designs with binary outcomes, --p1, --p2, --cluster-size, and --icc are required.")
                raise typer.Exit(1)
            
            result = sample_size_binary_cluster_rct(
                p1=p1,
                p2=p2,
                icc=icc,
                cluster_size=cluster_size,
                power=power,
                alpha=alpha
            )
            method_name = "sample_size_binary_cluster_rct"
        
        # Stepped wedge design
        elif design == DesignType.STEPPED_WEDGE:
            if not all([steps, individuals_per_cluster, icc, treatment_effect, std_dev]):
                console.print("[bold red]Error:[/bold red] For stepped wedge design, --steps, --individuals-per-cluster, --icc, --treatment-effect, and --std-dev are required.")
                raise typer.Exit(1)
            
            # For stepped wedge, we can do a search for number of clusters that gives desired power
            power_threshold = power
            min_clusters = 4
            max_clusters = 50
            
            for clusters in range(min_clusters, max_clusters + 1):
                sim_result = simulate_stepped_wedge(
                    clusters=clusters,
                    steps=steps,
                    individuals_per_cluster=individuals_per_cluster,
                    icc=icc,
                    treatment_effect=treatment_effect,
                    std_dev=std_dev,
                    nsim=nsim,
                    alpha=alpha
                )
                
                if sim_result["power"] >= power_threshold:
                    result = {
                        "clusters": clusters,
                        "steps": steps,
                        "individuals_per_cluster": individuals_per_cluster,
                        "total_n": clusters * steps * individuals_per_cluster,
                        "power": sim_result["power"],
                        "parameters": {
                            "clusters": clusters,
                            "steps": steps,
                            "individuals_per_cluster": individuals_per_cluster,
                            "icc": icc,
                            "treatment_effect": treatment_effect,
                            "std_dev": std_dev,
                            "power_threshold": power_threshold,
                            "alpha": alpha,
                            "nsim": nsim
                        }
                    }
                    method_name = "simulate_stepped_wedge"
                    break
            
            if result is None:
                console.print(f"[bold yellow]Warning:[/bold yellow] Could not achieve power of {power} with up to {max_clusters} clusters.")
                # Return the result for max_clusters
                sim_result = simulate_stepped_wedge(
                    clusters=max_clusters,
                    steps=steps,
                    individuals_per_cluster=individuals_per_cluster,
                    icc=icc,
                    treatment_effect=treatment_effect,
                    std_dev=std_dev,
                    nsim=nsim,
                    alpha=alpha
                )
                result = {
                    "clusters": max_clusters,
                    "steps": steps,
                    "individuals_per_cluster": individuals_per_cluster,
                    "total_n": max_clusters * steps * individuals_per_cluster,
                    "power": sim_result["power"],
                    "parameters": {
                        "clusters": max_clusters,
                        "steps": steps,
                        "individuals_per_cluster": individuals_per_cluster,
                        "icc": icc,
                        "treatment_effect": treatment_effect,
                        "std_dev": std_dev,
                        "power_threshold": power_threshold,
                        "alpha": alpha,
                        "nsim": nsim
                    }
                }
                method_name = "simulate_stepped_wedge"
        
        else:
            console.print(f"[bold red]Error:[/bold red] Sample size calculation not implemented for {design.value} design with {outcome.value} outcome.")
            raise typer.Exit(1)
        
        # Display result
        if result:
            if output_json:
                console.print(json.dumps(result, indent=2))
            else:
                display_result(result, method_name)
        else:
            console.print("[bold red]Error:[/bold red] Calculation failed.")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def power(
    design: DesignType = typer.Option(DesignType.PARALLEL, help="Study design"),
    outcome: OutcomeType = typer.Option(OutcomeType.CONTINUOUS, help="Outcome type"),
    n1: Optional[int] = typer.Option(None, help="Sample size for group 1 (for parallel design)"),
    n2: Optional[int] = typer.Option(None, help="Sample size for group 2 (for parallel design)"),
    n_clusters: Optional[int] = typer.Option(None, help="Number of clusters per arm (for cluster design)"),
    cluster_size: Optional[int] = typer.Option(None, help="Average number of individuals per cluster (for cluster designs)"),
    clusters: Optional[int] = typer.Option(None, help="Total number of clusters (for stepped wedge)"),
    steps: Optional[int] = typer.Option(None, help="Number of time steps (for stepped wedge design)"),
    individuals_per_cluster: Optional[int] = typer.Option(None, help="Number of individuals per cluster per step (for stepped wedge)"),
    delta: Optional[float] = typer.Option(None, help="Effect size (difference in means for continuous outcomes)"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation (for continuous outcomes)"),
    p1: Optional[float] = typer.Option(None, help="Proportion in control group (for binary outcomes)"),
    p2: Optional[float] = typer.Option(None, help="Proportion in intervention group (for binary outcomes)"),
    treatment_effect: Optional[float] = typer.Option(None, help="Effect size of intervention (for stepped wedge)"),
    icc: Optional[float] = typer.Option(None, help="Intracluster correlation coefficient (for cluster designs)"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    nsim: int = typer.Option(1000, help="Number of simulations (for simulation-based estimation)"),
    simulate: bool = typer.Option(False, help="Use simulation-based estimation instead of analytical formula"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """
    Calculate statistical power for a given study design and sample size.
    """
    try:
        result = None
        method_name = None
        
        # Parallel design with continuous outcome
        if design == DesignType.PARALLEL and outcome == OutcomeType.CONTINUOUS:
            if n1 is None or n2 is None or delta is None or std_dev is None:
                console.print("[bold red]Error:[/bold red] For continuous outcomes in parallel design, --n1, --n2, --delta, and --std-dev are required.")
                raise typer.Exit(1)
            
            if simulate:
                result = simulate_parallel_rct(
                    n1=n1,
                    n2=n2,
                    mean1=0,
                    mean2=delta,
                    std_dev=std_dev,
                    nsim=nsim,
                    alpha=alpha
                )
                method_name = "simulate_parallel_rct"
            else:
                result = power_difference_in_means(
                    n1=n1,
                    n2=n2,
                    delta=delta,
                    std_dev=std_dev,
                    alpha=alpha
                )
                method_name = "power_difference_in_means"
        
        # Cluster design with binary outcome
        elif design == DesignType.CLUSTER and outcome == OutcomeType.BINARY:
            if n_clusters is None or cluster_size is None or p1 is None or p2 is None or icc is None:
                console.print("[bold red]Error:[/bold red] For binary outcomes in cluster design, --n-clusters, --cluster-size, --p1, --p2, and --icc are required.")
                raise typer.Exit(1)
            
            if simulate:
                result = simulate_binary_cluster_rct(
                    n_clusters=n_clusters,
                    cluster_size=cluster_size,
                    icc=icc,
                    p1=p1,
                    p2=p2,
                    nsim=nsim,
                    alpha=alpha
                )
                method_name = "simulate_binary_cluster_rct"
            else:
                result = power_binary_cluster_rct(
                    n_clusters=n_clusters,
                    cluster_size=cluster_size,
                    icc=icc,
                    p1=p1,
                    p2=p2,
                    alpha=alpha
                )
                method_name = "power_binary_cluster_rct"
        
        # Stepped wedge design
        elif design == DesignType.STEPPED_WEDGE:
            if not all([clusters, steps, individuals_per_cluster, icc, treatment_effect, std_dev]):
                console.print("[bold red]Error:[/bold red] For stepped wedge design, --clusters, --steps, --individuals-per-cluster, --icc, --treatment-effect, and --std-dev are required.")
                raise typer.Exit(1)
            
            result = simulate_stepped_wedge(
                clusters=clusters,
                steps=steps,
                individuals_per_cluster=individuals_per_cluster,
                icc=icc,
                treatment_effect=treatment_effect,
                std_dev=std_dev,
                nsim=nsim,
                alpha=alpha
            )
            method_name = "simulate_stepped_wedge"
        
        else:
            console.print(f"[bold red]Error:[/bold red] Power calculation not implemented for {design.value} design with {outcome.value} outcome.")
            raise typer.Exit(1)
        
        # Display result
        if result:
            if output_json:
                console.print(json.dumps(result, indent=2))
            else:
                display_result(result, method_name)
        else:
            console.print("[bold red]Error:[/bold red] Calculation failed.")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def mde(
    design: DesignType = typer.Option(DesignType.PARALLEL, help="Study design"),
    outcome: OutcomeType = typer.Option(OutcomeType.CONTINUOUS, help="Outcome type"),
    n1: Optional[int] = typer.Option(None, help="Sample size for group 1 (for parallel design)"),
    n2: Optional[int] = typer.Option(None, help="Sample size for group 2 (for parallel design)"),
    n_clusters: Optional[int] = typer.Option(None, help="Number of clusters per arm (for cluster design)"),
    cluster_size: Optional[int] = typer.Option(None, help="Average number of individuals per cluster (for cluster designs)"),
    clusters: Optional[int] = typer.Option(None, help="Total number of clusters (for stepped wedge)"),
    steps: Optional[int] = typer.Option(None, help="Number of time steps (for stepped wedge design)"),
    individuals_per_cluster: Optional[int] = typer.Option(None, help="Number of individuals per cluster per step (for stepped wedge)"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation (for continuous outcomes)"),
    p1: Optional[float] = typer.Option(None, help="Baseline proportion (for binary outcomes)"),
    icc: Optional[float] = typer.Option(None, help="Intracluster correlation coefficient (for cluster designs)"),
    power: float = typer.Option(0.8, help="Desired statistical power"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    nsim: int = typer.Option(1000, help="Number of simulations (for simulation-based estimation)"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """
    Calculate minimum detectable effect for a given study design and sample size.
    """
    try:
        result = None
        method_name = None
        
        # Currently only implemented for cluster design with binary outcome
        if design == DesignType.CLUSTER and outcome == OutcomeType.BINARY:
            if n_clusters is None or cluster_size is None or p1 is None or icc is None:
                console.print("[bold red]Error:[/bold red] For binary outcomes in cluster design, --n-clusters, --cluster-size, --p1, and --icc are required.")
                raise typer.Exit(1)
            
            result = min_detectable_effect_binary_cluster_rct(
                n_clusters=n_clusters,
                cluster_size=cluster_size,
                icc=icc,
                p1=p1,
                power=power,
                alpha=alpha
            )
            method_name = "min_detectable_effect_binary_cluster_rct"
        
        else:
            console.print(f"[bold red]Error:[/bold red] MDE calculation not implemented for {design.value} design with {outcome.value} outcome.")
            raise typer.Exit(1)
        
        # Display result
        if result:
            if output_json:
                console.print(json.dumps(result, indent=2))
            else:
                display_result(result, method_name)
        else:
            console.print("[bold red]Error:[/bold red] Calculation failed.")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command("julia")
def julia_example():
    """
    Show example of how to use Julia for high-performance simulation.
    """
    try:
        from julia import Main
        
        # Check if the Julia file exists
        import os
        julia_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "julia_backend", "stepped_wedge.jl")
        
        if not os.path.exists(julia_file):
            console.print(f"[bold red]Error:[/bold red] Julia file not found at {julia_file}")
            console.print("Please ensure the Julia backend is correctly installed.")
            raise typer.Exit(1)
        
        # Initialize Julia
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        
        # Include the Julia file
        Main.include(julia_file)
        
        # Call the example function
        Main.call_from_python_example()
        
        console.print("\n[bold green]Success![/bold green] Julia is correctly configured.")
    
    except ImportError:
        console.print("[bold red]Error:[/bold red] Julia or PyJulia is not installed.")
        console.print("Please install Julia and PyJulia to use the high-performance simulation capabilities.")
        console.print("\nInstallation instructions:")
        console.print("1. Install Julia from https://julialang.org/downloads/")
        console.print("2. Install PyJulia: pip install julia")
        console.print("3. Run Python and execute:")
        console.print("   >>> import julia")
        console.print("   >>> julia.install()")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command("generate-script")
def generate_script(
    design: DesignType = typer.Option(DesignType.PARALLEL, help="Study design"),
    outcome: OutcomeType = typer.Option(OutcomeType.CONTINUOUS, help="Outcome type"),
    calculation: CalculationType = typer.Option(CalculationType.SAMPLE_SIZE, help="Type of calculation"),
    
    # Core parameters
    delta: Optional[float] = typer.Option(None, help="Minimum detectable effect (difference in means)"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation (for continuous outcomes)"),
    p1: Optional[float] = typer.Option(None, help="Proportion in control group (for binary outcomes)"),
    p2: Optional[float] = typer.Option(None, help="Proportion in intervention group (for binary outcomes)"),
    power: float = typer.Option(0.8, help="Desired statistical power (1 - beta)"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    allocation_ratio: float = typer.Option(1.0, help="Ratio of sample sizes (n2/n1) for parallel design"),
    
    # Sample size inputs (for power/MDE calculations)
    n1: Optional[int] = typer.Option(None, help="Sample size group 1 (for power/MDE calculations)"),
    n2: Optional[int] = typer.Option(None, help="Sample size group 2 (for power/MDE calculations)"),
    
    # Cluster design parameters
    cluster_size: Optional[int] = typer.Option(None, help="Average number of individuals per cluster (for cluster designs)"),
    icc: Optional[float] = typer.Option(None, help="Intracluster correlation coefficient (for cluster designs)"),
    n_clusters: Optional[int] = typer.Option(None, help="Number of clusters per arm (for cluster designs)"),
    
    # Method selection
    method: str = typer.Option("analytical", help="Calculation method (analytical, simulation, permutation)"),
    nsim: int = typer.Option(1000, help="Number of simulations (for simulation-based estimation)"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    
    # Output options
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file name (defaults to stdout)")
):
    """
    Generate a reproducible Python script for the specified analysis.
    
    This command creates a standalone Python script that reproduces the exact
    calculation with all parameters explicitly specified.
    """
    try:
        # Build parameters dictionary
        params = {
            'calculation_type': calculation.value.replace("-", " ").title(),
            'method': method,
            'alpha': alpha,
            'power': power,
            'allocation_ratio': allocation_ratio,
            'nsim': nsim,
            'seed': seed
        }
        
        # Add design-specific parameters
        if design == DesignType.PARALLEL:
            if outcome == OutcomeType.CONTINUOUS:
                # Validate required parameters
                if calculation == CalculationType.SAMPLE_SIZE:
                    if delta is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For sample size calculation, --delta and --std-dev are required.")
                        raise typer.Exit(1)
                    params.update({'mean1': 0.0, 'mean2': delta, 'std_dev': std_dev})
                elif calculation == CalculationType.POWER:
                    if n1 is None or n2 is None or delta is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For power calculation, --n1, --n2, --delta, and --std-dev are required.")
                        raise typer.Exit(1)
                    params.update({'n1': n1, 'n2': n2, 'mean1': 0.0, 'mean2': delta, 'std_dev': std_dev})
                elif calculation == CalculationType.MDE:
                    if n1 is None or n2 is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For MDE calculation, --n1, --n2, and --std-dev are required.")
                        raise typer.Exit(1)
                    params.update({'n1': n1, 'n2': n2, 'std_dev': std_dev})
                
                # Generate script using parallel RCT functions
                from app.components.parallel_rct import generate_cli_code_parallel_continuous
                script = generate_cli_code_parallel_continuous(params)
                
            elif outcome == OutcomeType.BINARY:
                # Validate required parameters
                if calculation == CalculationType.SAMPLE_SIZE:
                    if p1 is None or p2 is None:
                        console.print("[bold red]Error:[/bold red] For sample size calculation, --p1 and --p2 are required.")
                        raise typer.Exit(1)
                    params.update({'p1': p1, 'p2': p2})
                elif calculation == CalculationType.POWER:
                    if n1 is None or n2 is None or p1 is None or p2 is None:
                        console.print("[bold red]Error:[/bold red] For power calculation, --n1, --n2, --p1, and --p2 are required.")
                        raise typer.Exit(1)
                    params.update({'n1': n1, 'n2': n2, 'p1': p1, 'p2': p2})
                elif calculation == CalculationType.MDE:
                    if n1 is None or n2 is None or p1 is None:
                        console.print("[bold red]Error:[/bold red] For MDE calculation, --n1, --n2, and --p1 are required.")
                        raise typer.Exit(1)
                    params.update({'n1': n1, 'n2': n2, 'p1': p1})
                
                # Generate script using parallel RCT functions
                from app.components.parallel_rct import generate_cli_code_parallel_binary
                script = generate_cli_code_parallel_binary(params)
                
        elif design == DesignType.CLUSTER:
            if outcome == OutcomeType.CONTINUOUS:
                # Validate required cluster parameters
                if cluster_size is None or icc is None:
                    console.print("[bold red]Error:[/bold red] For cluster designs, --cluster-size and --icc are required.")
                    raise typer.Exit(1)
                
                if calculation == CalculationType.SAMPLE_SIZE:
                    if delta is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For sample size calculation, --delta and --std-dev are required.")
                        raise typer.Exit(1)
                    params.update({
                        'cluster_size': cluster_size, 'icc': icc,
                        'mean1': 0.0, 'mean2': delta, 'std_dev': std_dev
                    })
                elif calculation == CalculationType.POWER:
                    if n_clusters is None or delta is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For power calculation, --n-clusters, --delta, and --std-dev are required.")
                        raise typer.Exit(1)
                    params.update({
                        'n_clusters': n_clusters, 'cluster_size': cluster_size, 'icc': icc,
                        'mean1': 0.0, 'mean2': delta, 'std_dev': std_dev
                    })
                elif calculation == CalculationType.MDE:
                    if n_clusters is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For MDE calculation, --n-clusters and --std-dev are required.")
                        raise typer.Exit(1)
                    params.update({
                        'n_clusters': n_clusters, 'cluster_size': cluster_size, 'icc': icc,
                        'std_dev': std_dev
                    })
                
                # Generate script using cluster RCT functions
                from app.components.cluster_rct import generate_cli_code_cluster_continuous
                script = generate_cli_code_cluster_continuous(params)
                
            elif outcome == OutcomeType.BINARY:
                # Validate required cluster parameters
                if cluster_size is None or icc is None:
                    console.print("[bold red]Error:[/bold red] For cluster designs, --cluster-size and --icc are required.")
                    raise typer.Exit(1)
                
                if calculation == CalculationType.SAMPLE_SIZE:
                    if p1 is None or p2 is None:
                        console.print("[bold red]Error:[/bold red] For sample size calculation, --p1 and --p2 are required.")
                        raise typer.Exit(1)
                    params.update({
                        'cluster_size': cluster_size, 'icc': icc,
                        'p1': p1, 'p2': p2
                    })
                elif calculation == CalculationType.POWER:
                    if n_clusters is None or p1 is None or p2 is None:
                        console.print("[bold red]Error:[/bold red] For power calculation, --n-clusters, --p1, and --p2 are required.")
                        raise typer.Exit(1)
                    params.update({
                        'n_clusters': n_clusters, 'cluster_size': cluster_size, 'icc': icc,
                        'p1': p1, 'p2': p2
                    })
                elif calculation == CalculationType.MDE:
                    if n_clusters is None or p1 is None:
                        console.print("[bold red]Error:[/bold red] For MDE calculation, --n-clusters and --p1 are required.")
                        raise typer.Exit(1)
                    params.update({
                        'n_clusters': n_clusters, 'cluster_size': cluster_size, 'icc': icc,
                        'p1': p1
                    })
                
                # Generate script using cluster RCT functions
                from app.components.cluster_rct import generate_cli_code_cluster_binary
                script = generate_cli_code_cluster_binary(params)
        
        # Output the script
        if output_file:
            with open(output_file, 'w') as f:
                f.write(script)
            console.print(f"[bold green]✓[/bold green] Reproducible script saved to: {output_file}")
            console.print(f"[bold blue]Usage:[/bold blue] python {output_file}")
        else:
            console.print(script)
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command("single-arm")
def single_arm(
    outcome: OutcomeType = typer.Option(OutcomeType.BINARY, help="Outcome type (continuous or binary)"),
    calculation: CalculationType = typer.Option(CalculationType.SAMPLE_SIZE, help="Type of calculation"),
    
    # Common parameters
    alpha: float = typer.Option(0.05, help="Significance level"),
    power: float = typer.Option(0.8, help="Desired statistical power (1 - beta)"),
    
    # Continuous outcome parameters
    mean: Optional[float] = typer.Option(None, help="Sample mean (for continuous outcomes)"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation (for continuous outcomes)"),
    null_mean: Optional[float] = typer.Option(0.0, help="Null hypothesis mean (for continuous outcomes)"),
    
    # Binary outcome parameters  
    p: Optional[float] = typer.Option(None, help="Expected proportion (for binary outcomes)"),
    p0: Optional[float] = typer.Option(None, help="Null hypothesis proportion (for binary outcomes)"),
    design_method: str = typer.Option("standard", help="Design method: standard, ahern, or simons"),
    simon_type: str = typer.Option("optimal", help="Simon's design type: optimal or minimax"),
    
    # Sample size (for power calculations)
    n: Optional[int] = typer.Option(None, help="Sample size (required for power calculations)"),
    
    # Simon's two-stage specific parameters (for power calculations)
    n1: Optional[int] = typer.Option(None, help="Stage 1 sample size (for Simon's power calculation)"),
    r1: Optional[int] = typer.Option(None, help="Stage 1 rejection threshold (for Simon's power calculation)"),
    r: Optional[int] = typer.Option(None, help="Final rejection threshold (for power calculations)"),
    
    # Output options
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    generate_script: bool = typer.Option(False, "--script", help="Generate reproducible Python script"),
    script_output: Optional[str] = typer.Option(None, "--script-file", help="Save generated script to file")
):
    """
    Calculate sample size or power for single-arm (one-sample) studies.
    
    Examples:
    
    # Binary outcome with A'Hern design
    designpower single-arm --outcome binary --p 0.3 --p0 0.1 --design-method ahern
    
    # Binary outcome with Simon's two-stage design  
    designpower single-arm --outcome binary --p 0.4 --p0 0.2 --design-method simons
    
    # Continuous outcome
    designpower single-arm --outcome continuous --mean 1.5 --std-dev 2.0 --null-mean 0.0
    
    # Power calculation for standard binary design
    designpower single-arm --calculation power --outcome binary --n 50 --p 0.3 --p0 0.1
    """
    try:
        if generate_script:
            # Build parameters for script generation
            params = {
                'calculation_type': calculation.value.replace('-', ' ').title(),
                'alpha': alpha,
                'power': power
            }
            
            if outcome == OutcomeType.CONTINUOUS:
                if calculation == CalculationType.SAMPLE_SIZE and (mean is None or std_dev is None):
                    console.print("[bold red]Error:[/bold red] For continuous outcomes, --mean and --std-dev are required.")
                    raise typer.Exit(1)
                if calculation == CalculationType.POWER and n is None:
                    console.print("[bold red]Error:[/bold red] For power calculation, --n is required.")
                    raise typer.Exit(1)
                    
                params.update({
                    'mean': mean or 0.0,
                    'std_dev': std_dev or 1.0,
                    'null_mean': null_mean,
                    'n': n
                })
                
                from app.components.single_arm import generate_cli_code_single_arm_continuous
                script = generate_cli_code_single_arm_continuous(params)
                
            else:  # Binary outcomes
                if p is None or p0 is None:
                    console.print("[bold red]Error:[/bold red] For binary outcomes, --p and --p0 are required.")
                    raise typer.Exit(1)
                if calculation == CalculationType.POWER and n is None:
                    console.print("[bold red]Error:[/bold red] For power calculation, --n is required.")
                    raise typer.Exit(1)
                    
                params.update({
                    'p': p,
                    'p0': p0,
                    'design_method': design_method,
                    'simon_design_type': simon_type,
                    'n': n,
                    'n1': n1,
                    'r1': r1,
                    'r': r
                })
                
                from app.components.single_arm import generate_cli_code_single_arm_binary
                script = generate_cli_code_single_arm_binary(params)
            
            # Output the script
            if script_output:
                with open(script_output, 'w') as f:
                    f.write(script)
                console.print(f"[bold green]✓[/bold green] Reproducible script saved to: {script_output}")
                console.print(f"[bold blue]Usage:[/bold blue] python {script_output}")
            else:
                console.print(script)
                
        else:
            # Perform actual calculation
            if outcome == OutcomeType.CONTINUOUS:
                if calculation == CalculationType.SAMPLE_SIZE:
                    if mean is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For continuous outcomes, --mean and --std-dev are required.")
                        raise typer.Exit(1)
                    
                    result = one_sample_t_test_sample_size(
                        mean=mean,
                        null_mean=null_mean,
                        std_dev=std_dev,
                        alpha=alpha,
                        power=power
                    )
                    
                elif calculation == CalculationType.POWER:
                    if n is None or mean is None or std_dev is None:
                        console.print("[bold red]Error:[/bold red] For power calculation, --n, --mean, and --std-dev are required.")
                        raise typer.Exit(1)
                        
                    result = one_sample_t_test_power(
                        n=n,
                        mean=mean,
                        null_mean=null_mean,
                        std_dev=std_dev,
                        alpha=alpha
                    )
                    
                else:  # MDE
                    console.print("[bold red]Error:[/bold red] MDE calculation not yet implemented for single-arm continuous outcomes.")
                    raise typer.Exit(1)
                    
            else:  # Binary outcomes
                if p is None or p0 is None:
                    console.print("[bold red]Error:[/bold red] For binary outcomes, --p and --p0 are required.")
                    raise typer.Exit(1)
                    
                if calculation == CalculationType.SAMPLE_SIZE:
                    if design_method.lower() == "ahern":
                        result = ahern_sample_size(
                            p0=p0,
                            p1=p,
                            alpha=alpha,
                            beta=1-power
                        )
                        
                    elif design_method.lower() == "simons":
                        result = simons_two_stage_design(
                            p0=p0,
                            p1=p,
                            alpha=alpha,
                            beta=1-power,
                            design_type=simon_type.lower()
                        )
                        
                    else:  # Standard
                        sample_size = one_sample_proportion_test_sample_size(
                            p0=p0,
                            p1=p,
                            alpha=alpha,
                            power=power
                        )
                        result = {"n": sample_size}
                        
                elif calculation == CalculationType.POWER:
                    if n is None:
                        console.print("[bold red]Error:[/bold red] For power calculation, --n is required.")
                        raise typer.Exit(1)
                        
                    if design_method.lower() == "ahern":
                        if r is None:
                            console.print("[bold red]Error:[/bold red] For A'Hern power calculation, --r (rejection threshold) is required.")
                            raise typer.Exit(1)
                        result = ahern_power(n=n, r=r, p0=p0, p1=p)
                        
                    elif design_method.lower() == "simons":
                        if n1 is None or r1 is None or r is None:
                            console.print("[bold red]Error:[/bold red] For Simon's power calculation, --n1, --r1, and --r are required.")
                            raise typer.Exit(1)
                        power_val = simons_power(n1=n1, r1=r1, n=n, r=r, p=p)
                        result = {"power": power_val}
                        
                    else:  # Standard
                        power_val = one_sample_proportion_test_power(
                            n=n,
                            p0=p0,
                            p1=p,
                            alpha=alpha
                        )
                        result = {"power": power_val}
                        
                else:  # MDE
                    console.print("[bold red]Error:[/bold red] MDE calculation not yet implemented for single-arm binary outcomes.")
                    raise typer.Exit(1)
            
            # Display results
            if output_json:
                console.print(json.dumps(result, indent=2))
            else:
                display_result(result, f"Single-arm {outcome.value} ({design_method})")
                
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command("survival")
def survival(
    design: DesignType = typer.Option(DesignType.PARALLEL, help="Study design (parallel or single-arm)"),
    calculation: CalculationType = typer.Option(CalculationType.SAMPLE_SIZE, help="Type of calculation"),
    hypothesis: str = typer.Option("superiority", help="Hypothesis type: superiority or non-inferiority"),
    
    # Common parameters
    alpha: float = typer.Option(0.05, help="Significance level"),
    power: float = typer.Option(0.8, help="Desired statistical power (1 - beta)"),
    sides: int = typer.Option(2, help="One-sided (1) or two-sided (2) test"),
    
    # Survival-specific parameters
    median_control: Optional[float] = typer.Option(None, help="Median survival in control group (months)"),
    median_treatment: Optional[float] = typer.Option(None, help="Median survival in treatment group (months)"),
    hazard_ratio: Optional[float] = typer.Option(None, help="Hazard ratio (treatment/control)"),
    
    # For single-arm studies  
    median_null: Optional[float] = typer.Option(None, help="Null hypothesis median survival (single-arm)"),
    median_alt: Optional[float] = typer.Option(None, help="Alternative hypothesis median survival (single-arm)"),
    
    # Study design parameters
    enrollment_period: float = typer.Option(12.0, help="Enrollment period (months)"),
    follow_up_period: float = typer.Option(12.0, help="Follow-up period (months)"),
    dropout_rate: float = typer.Option(0.1, help="Expected dropout rate"),
    allocation_ratio: float = typer.Option(1.0, help="Allocation ratio (n2/n1) for parallel design"),
    
    # Non-inferiority parameters
    ni_margin: Optional[float] = typer.Option(None, help="Non-inferiority margin (hazard ratio)"),
    assumed_hr: Optional[float] = typer.Option(1.0, help="Assumed true HR for non-inferiority"),
    
    # Sample sizes (for power calculations)
    n: Optional[int] = typer.Option(None, help="Total sample size (single-arm)"),
    n1: Optional[int] = typer.Option(None, help="Sample size group 1 (parallel)"),
    n2: Optional[int] = typer.Option(None, help="Sample size group 2 (parallel)"),
    
    # Output options
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    generate_script: bool = typer.Option(False, "--script", help="Generate reproducible Python script"),
    script_output: Optional[str] = typer.Option(None, "--script-file", help="Save generated script to file")
):
    """
    Calculate sample size or power for survival analysis studies.
    
    Examples:
    
    # Parallel survival study (superiority)
    designpower survival --design parallel --median-control 12 --median-treatment 18
    
    # Parallel study with hazard ratio
    designpower survival --design parallel --median-control 12 --hazard-ratio 0.7
    
    # Single-arm survival study
    designpower survival --design single-arm --median-null 6 --median-alt 12
    
    # Non-inferiority survival study
    designpower survival --hypothesis non-inferiority --median-control 12 --ni-margin 1.3
    
    # Power calculation
    designpower survival --calculation power --n1 100 --n2 100 --median-control 12 --median-treatment 18
    """
    try:
        if generate_script:
            # Build parameters for script generation
            params = {
                'calculation_type': calculation.value.replace('-', ' ').title(),
                'design': design.value,
                'hypothesis_type': hypothesis.title(),
                'alpha': alpha,
                'power': power,
                'sides': sides,
                'enrollment_period': enrollment_period,
                'follow_up_period': follow_up_period,
                'dropout_rate': dropout_rate,
                'allocation_ratio': allocation_ratio
            }
            
            if design == DesignType.PARALLEL:
                if median_control is None:
                    console.print("[bold red]Error:[/bold red] For parallel studies, --median-control is required.")
                    raise typer.Exit(1)
                    
                # Determine median_treatment from hazard_ratio if provided
                if hazard_ratio is not None and median_treatment is None:
                    median_treatment = median_control / hazard_ratio
                elif median_treatment is None:
                    console.print("[bold red]Error:[/bold red] Either --median-treatment or --hazard-ratio is required.")
                    raise typer.Exit(1)
                    
                params.update({
                    'median_control': median_control,
                    'median_treatment': median_treatment,
                    'hazard_ratio': hazard_ratio,
                    'ni_margin': ni_margin,
                    'assumed_hr': assumed_hr,
                    'n1': n1,
                    'n2': n2
                })
                
                from app.components.parallel_rct import generate_cli_code_parallel_survival
                script = generate_cli_code_parallel_survival(params)
                
            else:  # Single-arm
                if median_null is None or median_alt is None:
                    console.print("[bold red]Error:[/bold red] For single-arm studies, --median-null and --median-alt are required.")
                    raise typer.Exit(1)
                    
                params.update({
                    'median_null': median_null,
                    'median_alt': median_alt,
                    'n': n
                })
                
                from app.components.single_arm import generate_cli_code_single_arm_survival
                script = generate_cli_code_single_arm_survival(params)
            
            # Output the script
            if script_output:
                with open(script_output, 'w') as f:
                    f.write(script)
                console.print(f"[bold green]✓[/bold green] Reproducible script saved to: {script_output}")
                console.print(f"[bold blue]Usage:[/bold blue] python {script_output}")
            else:
                console.print(script)
                
        else:
            # Perform actual calculation
            if design == DesignType.PARALLEL:
                if median_control is None:
                    console.print("[bold red]Error:[/bold red] For parallel studies, --median-control is required.")
                    raise typer.Exit(1)
                    
                # Determine median_treatment from hazard_ratio if provided
                if hazard_ratio is not None and median_treatment is None:
                    median_treatment = median_control / hazard_ratio
                elif median_treatment is None:
                    console.print("[bold red]Error:[/bold red] Either --median-treatment or --hazard-ratio is required.")
                    raise typer.Exit(1)
                
                if hypothesis.lower() == "non-inferiority":
                    if ni_margin is None:
                        console.print("[bold red]Error:[/bold red] For non-inferiority, --ni-margin is required.")
                        raise typer.Exit(1)
                        
                    if calculation == CalculationType.SAMPLE_SIZE:
                        result = sample_size_survival_non_inferiority(
                            median1=median_control,
                            non_inferiority_margin=ni_margin,
                            enrollment_period=enrollment_period,
                            follow_up_period=follow_up_period,
                            dropout_rate=dropout_rate,
                            power=power,
                            alpha=alpha,
                            allocation_ratio=allocation_ratio,
                            assumed_hr=assumed_hr
                        )
                    elif calculation == CalculationType.POWER:
                        if n1 is None or n2 is None:
                            console.print("[bold red]Error:[/bold red] For power calculation, --n1 and --n2 are required.")
                            raise typer.Exit(1)
                        result = power_survival_non_inferiority(
                            n1=n1, n2=n2,
                            median1=median_control,
                            non_inferiority_margin=ni_margin,
                            enrollment_period=enrollment_period,
                            follow_up_period=follow_up_period,
                            dropout_rate=dropout_rate,
                            alpha=alpha,
                            assumed_hr=assumed_hr
                        )
                    else:
                        console.print("[bold red]Error:[/bold red] MDE calculation not implemented for non-inferiority survival studies.")
                        raise typer.Exit(1)
                        
                else:  # Superiority
                    if calculation == CalculationType.SAMPLE_SIZE:
                        result = sample_size_survival(
                            median1=median_control,
                            median2=median_treatment,
                            enrollment_period=enrollment_period,
                            follow_up_period=follow_up_period,
                            dropout_rate=dropout_rate,
                            power=power,
                            alpha=alpha,
                            allocation_ratio=allocation_ratio,
                            sides=sides
                        )
                    elif calculation == CalculationType.POWER:
                        if n1 is None or n2 is None:
                            console.print("[bold red]Error:[/bold red] For power calculation, --n1 and --n2 are required.")
                            raise typer.Exit(1)
                        result = power_survival(
                            n1=n1, n2=n2,
                            median1=median_control,
                            median2=median_treatment,
                            enrollment_period=enrollment_period,
                            follow_up_period=follow_up_period,
                            dropout_rate=dropout_rate,
                            alpha=alpha,
                            sides=sides
                        )
                    else:  # MDE
                        if n1 is None or n2 is None:
                            console.print("[bold red]Error:[/bold red] For MDE calculation, --n1 and --n2 are required.")
                            raise typer.Exit(1)
                        result = min_detectable_effect_survival(
                            n1=n1, n2=n2,
                            median1=median_control,
                            enrollment_period=enrollment_period,
                            follow_up_period=follow_up_period,
                            dropout_rate=dropout_rate,
                            alpha=alpha,
                            power=power,
                            sides=sides
                        )
                        
            else:  # Single-arm
                if median_null is None or median_alt is None:
                    console.print("[bold red]Error:[/bold red] For single-arm studies, --median-null and --median-alt are required.")
                    raise typer.Exit(1)
                    
                if calculation == CalculationType.SAMPLE_SIZE:
                    result = one_sample_survival_test_sample_size(
                        median_null=median_null,
                        median_alt=median_alt,
                        enrollment_period=enrollment_period,
                        follow_up_period=follow_up_period,
                        dropout_rate=dropout_rate,
                        alpha=alpha,
                        power=power,
                        sides=sides
                    )
                elif calculation == CalculationType.POWER:
                    if n is None:
                        console.print("[bold red]Error:[/bold red] For power calculation, --n is required.")
                        raise typer.Exit(1)
                    result = one_sample_survival_test_power(
                        n=n,
                        median_null=median_null,
                        median_alt=median_alt,
                        enrollment_period=enrollment_period,
                        follow_up_period=follow_up_period,
                        dropout_rate=dropout_rate,
                        alpha=alpha,
                        sides=sides
                    )
                else:
                    console.print("[bold red]Error:[/bold red] MDE calculation not yet implemented for single-arm survival studies.")
                    raise typer.Exit(1)
            
            # Display results
            if output_json:
                console.print(json.dumps(result, indent=2))
            else:
                display_result(result, f"{design.value.title()} survival ({hypothesis})")
                
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
