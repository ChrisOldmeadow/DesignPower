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


class OutcomeType(str, Enum):
    """Enum for supported outcome types."""
    CONTINUOUS = "continuous"
    BINARY = "binary"


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
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """
    Calculate required sample size for a given study design.
    """
    try:
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


if __name__ == "__main__":
    app()
