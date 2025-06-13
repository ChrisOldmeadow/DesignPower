"""
CLI commands for parallel group RCT designs.

This module contains commands for sample size, power, and MDE calculations
for parallel group randomized controlled trials.
"""
import typer
from typing import Optional
import json

from cli.shared import OutcomeType, console, display_result
from core.power import (
    sample_size_difference_in_means,
    power_difference_in_means
)

app = typer.Typer(help="Parallel Group RCT Commands")


@app.command()
def sample_size(
    outcome: OutcomeType = typer.Option(OutcomeType.CONTINUOUS, help="Outcome type"),
    delta: Optional[float] = typer.Option(None, help="Minimum detectable effect (difference in means)"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation (for continuous outcomes)"),
    p1: Optional[float] = typer.Option(None, help="Proportion in control group (for binary outcomes)"),
    p2: Optional[float] = typer.Option(None, help="Proportion in intervention group (for binary outcomes)"),
    power: float = typer.Option(0.8, help="Desired statistical power (1 - beta)"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    allocation_ratio: float = typer.Option(1.0, help="Ratio of sample sizes (n2/n1)"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    generate_script: bool = typer.Option(False, "--script", help="Generate reproducible Python script"),
    script_output: Optional[str] = typer.Option(None, "--script-file", help="Save generated script to file")
):
    """
    Calculate required sample size for parallel group RCT.
    """
    try:
        # If script generation is requested
        if generate_script:
            params = {
                'calculation_type': 'Sample Size',
                'method': 'analytical',
                'alpha': alpha,
                'power': power,
                'allocation_ratio': allocation_ratio
            }
            
            if outcome == OutcomeType.CONTINUOUS:
                if delta is None or std_dev is None:
                    console.print("[bold red]Error:[/bold red] For continuous outcomes, --delta and --std-dev are required.")
                    raise typer.Exit(1)
                params.update({'mean1': 0.0, 'mean2': delta, 'std_dev': std_dev})
                from app.components.parallel_rct.cli_generation import generate_cli_code_parallel_continuous
                script = generate_cli_code_parallel_continuous(params)
                
            elif outcome == OutcomeType.BINARY:
                if p1 is None or p2 is None:
                    console.print("[bold red]Error:[/bold red] For binary outcomes, --p1 and --p2 are required.")
                    raise typer.Exit(1)
                params.update({'p1': p1, 'p2': p2})
                from app.components.parallel_rct.cli_generation import generate_cli_code_parallel_binary
                script = generate_cli_code_parallel_binary(params)
            
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
        
        # Original calculation logic for basic results display
        if outcome == OutcomeType.CONTINUOUS:
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