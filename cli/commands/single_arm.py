"""
CLI commands for single-arm trial designs.

This module contains commands for sample size and power calculations
for single-arm trials with various design methods.
"""
import typer
from typing import Optional
import json

from cli.shared import OutcomeType, console, display_result
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

app = typer.Typer(help="Single-Arm Trial Commands")


@app.command()
def sample_size(
    design: str = typer.Option("standard", help="Design method: standard, ahern, simons"),
    outcome: OutcomeType = typer.Option(OutcomeType.CONTINUOUS, help="Outcome type"),
    mean: Optional[float] = typer.Option(None, help="Expected mean (for continuous outcomes)"),
    mean0: Optional[float] = typer.Option(None, help="Null hypothesis mean (for continuous outcomes)"),
    std_dev: Optional[float] = typer.Option(None, help="Standard deviation (for continuous outcomes)"),
    p: Optional[float] = typer.Option(None, help="Expected proportion (for binary outcomes)"),
    p0: Optional[float] = typer.Option(None, help="Null hypothesis proportion (for binary outcomes)"),
    power: float = typer.Option(0.8, help="Desired statistical power"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    simon_type: str = typer.Option("optimal", help="Simon's design type: optimal or minimax"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    generate_script: bool = typer.Option(False, "--script", help="Generate reproducible Python script"),
    script_output: Optional[str] = typer.Option(None, "--script-file", help="Save generated script to file")
):
    """
    Calculate required sample size for single-arm trial.
    """
    try:
        # If script generation is requested
        if generate_script:
            params = {
                'calculation_type': 'Sample Size',
                'design_method': design,
                'alpha': alpha,
                'power': power
            }
            
            if outcome == OutcomeType.CONTINUOUS:
                if mean is None or mean0 is None or std_dev is None:
                    console.print("[bold red]Error:[/bold red] For continuous outcomes, --mean, --mean0, and --std-dev are required.")
                    raise typer.Exit(1)
                params.update({'mean': mean, 'mean0': mean0, 'std_dev': std_dev})
                from app.components.single_arm import generate_cli_code_single_arm_continuous
                script = generate_cli_code_single_arm_continuous(params)
                
            elif outcome == OutcomeType.BINARY:
                if p is None or p0 is None:
                    console.print("[bold red]Error:[/bold red] For binary outcomes, --p and --p0 are required.")
                    raise typer.Exit(1)
                params.update({'p': p, 'p0': p0, 'simon_design_type': simon_type})
                from app.components.single_arm import generate_cli_code_single_arm_binary
                script = generate_cli_code_single_arm_binary(params)
            
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
        if outcome == OutcomeType.CONTINUOUS:
            if mean is None or mean0 is None or std_dev is None:
                console.print("[bold red]Error:[/bold red] For continuous outcomes, --mean, --mean0, and --std-dev are required.")
                raise typer.Exit(1)
            
            result = one_sample_t_test_sample_size(
                mean=mean,
                mean0=mean0,
                std_dev=std_dev,
                power=power,
                alpha=alpha
            )
            method_name = "one_sample_t_test_sample_size"
            
        elif outcome == OutcomeType.BINARY:
            if p is None or p0 is None:
                console.print("[bold red]Error:[/bold red] For binary outcomes, --p and --p0 are required.")
                raise typer.Exit(1)
            
            if design.lower() == "ahern":
                result = ahern_sample_size(
                    p0=p0,
                    p1=p,
                    alpha=alpha,
                    beta=1-power
                )
                method_name = "ahern_sample_size"
            elif design.lower() == "simons":
                result = simons_two_stage_design(
                    p0=p0,
                    p1=p,
                    alpha=alpha,
                    beta=1-power,
                    design_type=simon_type.lower()
                )
                method_name = "simons_two_stage_design"
            else:
                result = one_sample_proportion_test_sample_size(
                    p=p,
                    p0=p0,
                    power=power,
                    alpha=alpha
                )
                method_name = "one_sample_proportion_test_sample_size"
        
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