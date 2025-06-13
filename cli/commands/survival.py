"""
CLI commands for survival analysis.

This module contains commands for sample size and power calculations
for survival outcomes in various trial designs.
"""
import typer
from typing import Optional
import json

from cli.shared import console, display_result
from core.designs.parallel.analytical_survival import (
    sample_size_survival,
    power_survival,
    min_detectable_effect_survival
)
from core.designs.single_arm.survival import (
    one_sample_survival_test_sample_size,
    one_sample_survival_test_power
)

app = typer.Typer(help="Survival Analysis Commands")


@app.command()
def sample_size(
    design: str = typer.Option("parallel", help="Study design: parallel or single-arm"),
    median_control: Optional[float] = typer.Option(None, help="Median survival in control group (months)"),
    median_treatment: Optional[float] = typer.Option(None, help="Median survival in treatment group (months)"),
    median_null: Optional[float] = typer.Option(None, help="Null hypothesis median (for single-arm)"),
    enrollment_period: float = typer.Option(12.0, help="Enrollment period (months)"),
    follow_up_period: float = typer.Option(12.0, help="Follow-up period (months)"),
    dropout_rate: float = typer.Option(0.1, help="Dropout rate"),
    power: float = typer.Option(0.8, help="Desired statistical power"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    allocation_ratio: float = typer.Option(1.0, help="Allocation ratio (parallel design)"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """
    Calculate required sample size for survival analysis.
    """
    try:
        if design.lower() == "parallel":
            if median_control is None or median_treatment is None:
                console.print("[bold red]Error:[/bold red] For parallel design, --median-control and --median-treatment are required.")
                raise typer.Exit(1)
            
            result = sample_size_survival(
                median_control=median_control,
                median_treatment=median_treatment,
                enrollment_period=enrollment_period,
                follow_up_period=follow_up_period,
                dropout_rate=dropout_rate,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio
            )
            method_name = "sample_size_survival"
            
        elif design.lower() == "single-arm":
            if median_null is None or median_treatment is None:
                console.print("[bold red]Error:[/bold red] For single-arm design, --median-null and --median-treatment are required.")
                raise typer.Exit(1)
            
            result = one_sample_survival_test_sample_size(
                median_null=median_null,
                median_alt=median_treatment,
                enrollment_period=enrollment_period,
                follow_up_period=follow_up_period,
                dropout_rate=dropout_rate,
                power=power,
                alpha=alpha
            )
            method_name = "one_sample_survival_test_sample_size"
        
        else:
            console.print(f"[bold red]Error:[/bold red] Unknown design: {design}")
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