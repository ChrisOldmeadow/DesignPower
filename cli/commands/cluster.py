"""
CLI commands for cluster randomized trial designs.

This module contains commands for sample size, power, and MDE calculations
for cluster randomized controlled trials.
"""
import typer
from typing import Optional
import json

from cli.shared import OutcomeType, console, display_result
from core.power import (
    sample_size_binary_cluster_rct,
    power_binary_cluster_rct,
    min_detectable_effect_binary_cluster_rct
)

app = typer.Typer(help="Cluster Randomized Trial Commands")


@app.command()
def sample_size(
    outcome: OutcomeType = typer.Option(OutcomeType.BINARY, help="Outcome type"),
    cluster_size: int = typer.Option(..., help="Average number of individuals per cluster"),
    icc: float = typer.Option(..., help="Intracluster correlation coefficient"),
    delta: Optional[float] = typer.Option(None, help="Effect size (for continuous outcomes)"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation (for continuous outcomes)"),
    p1: Optional[float] = typer.Option(None, help="Proportion in control group (for binary outcomes)"),
    p2: Optional[float] = typer.Option(None, help="Proportion in intervention group (for binary outcomes)"),
    power: float = typer.Option(0.8, help="Desired statistical power"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    generate_script: bool = typer.Option(False, "--script", help="Generate reproducible Python script"),
    script_output: Optional[str] = typer.Option(None, "--script-file", help="Save generated script to file")
):
    """
    Calculate required number of clusters for cluster RCT.
    """
    try:
        # If script generation is requested
        if generate_script:
            params = {
                'calculation_type': 'Sample Size',
                'method': 'analytical',
                'alpha': alpha,
                'power': power,
                'cluster_size': cluster_size,
                'icc': icc
            }
            
            if outcome == OutcomeType.CONTINUOUS:
                if delta is None or std_dev is None:
                    console.print("[bold red]Error:[/bold red] For continuous outcomes, --delta and --std-dev are required.")
                    raise typer.Exit(1)
                params.update({'mean1': 0.0, 'mean2': delta, 'std_dev': std_dev})
                from app.components.cluster_rct.cli_generation import generate_cli_code_cluster_continuous
                script = generate_cli_code_cluster_continuous(params)
                
            elif outcome == OutcomeType.BINARY:
                if p1 is None or p2 is None:
                    console.print("[bold red]Error:[/bold red] For binary outcomes, --p1 and --p2 are required.")
                    raise typer.Exit(1)
                params.update({'p1': p1, 'p2': p2})
                from app.components.cluster_rct.cli_generation import generate_cli_code_cluster_binary
                script = generate_cli_code_cluster_binary(params)
            
            else:
                console.print(f"[bold red]Error:[/bold red] Script generation not yet supported for {outcome.value} outcome.")
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
        
        # Original calculation logic
        if outcome == OutcomeType.BINARY:
            if p1 is None or p2 is None:
                console.print("[bold red]Error:[/bold red] For binary outcomes, --p1 and --p2 are required.")
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
        
        else:
            console.print(f"[bold red]Error:[/bold red] Sample size calculation not implemented for {outcome.value} outcome.")
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