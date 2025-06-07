"""
Modern CLI for DesignPower that leverages dashboard components.

This CLI wraps the dashboard calculation functions to ensure feature parity
and consistent results between CLI and dashboard interfaces.
"""
import typer
from typing import Optional
from enum import Enum
from rich.console import Console
from rich.table import Table
import json

# Import dashboard calculation functions
from app.components.parallel_rct import (
    calculate_parallel_continuous,
    calculate_parallel_binary, 
    calculate_parallel_survival
)
from app.components.single_arm import (
    calculate_single_arm_continuous,
    calculate_single_arm_binary,
    calculate_single_arm_survival
)
from app.components.cluster_rct import (
    calculate_cluster_continuous,
    calculate_cluster_binary
)

app = typer.Typer(help="DesignPower CLI - Modern interface with full feature parity")
console = Console()


class DesignType(str, Enum):
    """Supported study designs."""
    PARALLEL = "parallel"
    SINGLE_ARM = "single-arm" 
    CLUSTER = "cluster"


class OutcomeType(str, Enum):
    """Supported outcome types."""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    SURVIVAL = "survival"


class CalculationType(str, Enum):
    """Calculation types."""
    SAMPLE_SIZE = "sample-size"
    POWER = "power"
    MDE = "mde"


class HypothesisType(str, Enum):
    """Hypothesis types."""
    SUPERIORITY = "superiority"
    NON_INFERIORITY = "non-inferiority"
    EQUIVALENCE = "equivalence"


def display_result(result: dict, title: str = "Results"):
    """Display calculation results in formatted tables."""
    
    # Main results table
    main_table = Table(title=title)
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", style="green")
    
    # Display key results (exclude nested dicts and parameters)
    for key, value in result.items():
        if key not in ["parameters", "inputs"] and not isinstance(value, dict):
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            main_table.add_row(key.replace("_", " ").title(), formatted_value)
    
    console.print(main_table)
    
    # Parameters table if available
    if "parameters" in result or "inputs" in result:
        param_data = result.get("parameters", result.get("inputs", {}))
        if param_data:
            param_table = Table(title="Input Parameters")
            param_table.add_column("Parameter", style="cyan")
            param_table.add_column("Value", style="green")
            
            for key, value in param_data.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                param_table.add_row(key.replace("_", " ").title(), formatted_value)
            
            console.print(param_table)


@app.command()
def calculate(
    design: DesignType = typer.Argument(..., help="Study design type"),
    outcome: OutcomeType = typer.Argument(..., help="Outcome type"),
    calculation: CalculationType = typer.Argument(..., help="Calculation type"),
    
    # Common parameters
    alpha: float = typer.Option(0.05, help="Significance level"),
    power: float = typer.Option(0.8, help="Statistical power (for sample size/MDE)"),
    
    # Sample sizes (for power/MDE calculations)
    n1: Optional[int] = typer.Option(None, help="Sample size group 1"),
    n2: Optional[int] = typer.Option(None, help="Sample size group 2 (parallel design)"),
    n: Optional[int] = typer.Option(None, help="Total sample size (single arm)"),
    
    # Effect parameters - Binary
    p1: Optional[float] = typer.Option(None, help="Proportion in control/baseline"),
    p2: Optional[float] = typer.Option(None, help="Proportion in intervention (parallel)"),
    p: Optional[float] = typer.Option(None, help="Expected proportion (single arm)"),
    p0: Optional[float] = typer.Option(None, help="Null hypothesis proportion"),
    
    # Effect parameters - Continuous  
    mean1: Optional[float] = typer.Option(None, help="Mean in control/baseline"),
    mean2: Optional[float] = typer.Option(None, help="Mean in intervention (parallel)"), 
    mean: Optional[float] = typer.Option(None, help="Expected mean (single arm)"),
    null_mean: Optional[float] = typer.Option(None, help="Null hypothesis mean"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation"),
    
    # Effect parameters - Survival
    median1: Optional[float] = typer.Option(None, help="Median survival control"),
    median2: Optional[float] = typer.Option(None, help="Median survival intervention"),
    median_survival: Optional[float] = typer.Option(None, help="Expected median survival"),
    null_median_survival: Optional[float] = typer.Option(None, help="Null median survival"),
    
    # Study design parameters
    accrual_time: Optional[float] = typer.Option(None, help="Accrual period (survival)"),
    follow_up_time: Optional[float] = typer.Option(None, help="Follow-up period (survival)"),
    dropout_rate: float = typer.Option(0.1, help="Dropout rate"),
    allocation_ratio: float = typer.Option(1.0, help="Allocation ratio (n2/n1)"),
    
    # Cluster design parameters
    n_clusters: Optional[int] = typer.Option(None, help="Number of clusters per arm"),
    cluster_size: Optional[int] = typer.Option(None, help="Average cluster size"),
    icc: Optional[float] = typer.Option(None, help="Intracluster correlation"),
    
    # Advanced options
    hypothesis: HypothesisType = typer.Option(HypothesisType.SUPERIORITY, help="Hypothesis type"),
    method: str = typer.Option("analytical", help="Method: analytical or simulation"),
    nsim: int = typer.Option(1000, help="Number of simulations"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
    
    # Non-inferiority specific
    nim: Optional[float] = typer.Option(None, help="Non-inferiority margin"),
    
    # Output options
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output")
):
    """
    Perform sample size, power, or MDE calculations for various study designs.
    
    Examples:
        # Parallel RCT sample size for binary outcome
        designpower calculate parallel binary sample-size --p1 0.3 --p2 0.5
        
        # Single arm power calculation
        designpower calculate single-arm continuous power --n 50 --mean 2.0 --null-mean 0 --std-dev 1.5
        
        # Cluster RCT with ICC
        designpower calculate cluster binary sample-size --p1 0.3 --p2 0.5 --cluster-size 20 --icc 0.05
    """
    try:
        # Build parameters dictionary
        params = {
            "calculation_type": calculation.value.replace("-", " ").title(),
            "calc_type": calculation.value.replace("-", " ").title(),  # For cluster designs
            "hypothesis_type": hypothesis.value.replace("-", " ").title(),
            "method": method,
            "alpha": alpha,
            "power": power,
            "allocation_ratio": allocation_ratio,
            "dropout_rate": dropout_rate,
            "use_simulation": method.lower() == "simulation",
            "nsim": nsim
        }
        
        if seed is not None:
            params["seed"] = seed
        if nim is not None:
            params["nim"] = nim
            
        # Add design-specific parameters
        if design == DesignType.PARALLEL:
            if n1 is not None:
                params["n1"] = n1
            if n2 is not None: 
                params["n2"] = n2
        elif design == DesignType.SINGLE_ARM:
            if n is not None:
                params["n"] = n
        elif design == DesignType.CLUSTER:
            if n_clusters is not None:
                params["n_clusters"] = n_clusters
            if cluster_size is not None:
                params["cluster_size"] = cluster_size
            if icc is not None:
                params["icc"] = icc
        
        # Add outcome-specific parameters
        if outcome == OutcomeType.BINARY:
            if p1 is not None:
                params["p1"] = p1
            if p2 is not None:
                params["p2"] = p2
            if p is not None:
                params["p"] = p  
            if p0 is not None:
                params["p0"] = p0
        elif outcome == OutcomeType.CONTINUOUS:
            if mean1 is not None:
                params["mean1"] = mean1
            if mean2 is not None:
                params["mean2"] = mean2
            if mean is not None:
                params["mean"] = mean
            if null_mean is not None:
                params["null_mean"] = null_mean
            if std_dev is not None:
                params["std_dev"] = std_dev
        elif outcome == OutcomeType.SURVIVAL:
            if median1 is not None:
                params["median1"] = median1
            if median2 is not None:
                params["median2"] = median2  
            if median_survival is not None:
                params["median_survival"] = median_survival
            if null_median_survival is not None:
                params["null_median_survival"] = null_median_survival
            if accrual_time is not None:
                params["accrual_time"] = accrual_time
            if follow_up_time is not None:
                params["follow_up_time"] = follow_up_time
        
        # Call appropriate calculation function
        result = None
        
        if design == DesignType.PARALLEL:
            if outcome == OutcomeType.CONTINUOUS:
                result = calculate_parallel_continuous(params)
            elif outcome == OutcomeType.BINARY:
                result = calculate_parallel_binary(params)
            elif outcome == OutcomeType.SURVIVAL:
                result = calculate_parallel_survival(params)
        elif design == DesignType.SINGLE_ARM:
            if outcome == OutcomeType.CONTINUOUS:
                result = calculate_single_arm_continuous(params)
            elif outcome == OutcomeType.BINARY:
                result = calculate_single_arm_binary(params)
            elif outcome == OutcomeType.SURVIVAL:
                result = calculate_single_arm_survival(params)
        elif design == DesignType.CLUSTER:
            if outcome == OutcomeType.CONTINUOUS:
                result = calculate_cluster_continuous(params)
            elif outcome == OutcomeType.BINARY:
                result = calculate_cluster_binary(params)
        
        if result is None:
            console.print(f"[bold red]Error:[/bold red] {design.value} design with {outcome.value} outcome not supported")
            raise typer.Exit(1)
        
        # Display results
        if output_json:
            console.print(json.dumps(result, indent=2, default=str))
        else:
            title = f"{design.value.title()} {outcome.value.title()} - {calculation.value.replace('-', ' ').title()}"
            display_result(result, title)
            
        if verbose:
            console.print(f"\n[dim]Parameters used: {params}[/dim]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def list_designs():
    """List all supported design and outcome combinations."""
    combinations = [
        ("parallel", "continuous", "✅"),
        ("parallel", "binary", "✅"), 
        ("parallel", "survival", "✅"),
        ("single-arm", "continuous", "✅"),
        ("single-arm", "binary", "✅"),
        ("single-arm", "survival", "✅"),
        ("cluster", "continuous", "✅"),
        ("cluster", "binary", "✅"),
        ("cluster", "survival", "❌"),
    ]
    
    table = Table(title="Supported Design/Outcome Combinations")
    table.add_column("Design", style="cyan")
    table.add_column("Outcome", style="green") 
    table.add_column("Status", style="yellow")
    
    for design, outcome, status in combinations:
        table.add_row(design, outcome, status)
    
    console.print(table)


if __name__ == "__main__":
    app()