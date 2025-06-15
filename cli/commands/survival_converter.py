"""
CLI commands for survival parameter conversion.

This module provides command-line interfaces for converting between different
survival analysis parameters including median survival, hazard rates, survival
fractions, and event rates.
"""

import typer
import json
from typing import Optional
from cli.shared import console
from core.utils.survival_converters import (
    convert_survival_parameters,
    convert_hazard_ratio_scenario,
    convert_survival_parameters_with_units
)

app = typer.Typer(help="Convert between different survival analysis parameters")


@app.command()
def convert(
    median: Optional[float] = typer.Option(None, help='Median survival time'),
    hazard: Optional[float] = typer.Option(None, help='Instantaneous hazard rate'),
    survival_fraction: Optional[float] = typer.Option(None, help='Survival fraction (0-1)'),
    event_rate: Optional[float] = typer.Option(None, help='Event rate (0-1)'),
    time_point: float = typer.Option(12.0, help='Time point for fractions'),
    time_unit: str = typer.Option('months', help='Time unit'),
    output_format: str = typer.Option('table', help='Output format (table/json/compact)')
):
    """
    Convert between survival parameters.
    
    Provide any one survival parameter to get all equivalent parameters.
    
    Examples:
        designpower survival-converter convert --median 12
        designpower survival-converter convert --survival-fraction 0.7 --time-point 24
        designpower survival-converter convert --hazard 0.058
    """
    # Count provided parameters
    provided_params = sum([
        median is not None,
        hazard is not None,
        survival_fraction is not None,
        event_rate is not None
    ])
    
    if provided_params == 0:
        console.print("Error: Provide at least one survival parameter", style="red")
        raise typer.Exit(1)
    
    if provided_params > 1:
        console.print("Note: Multiple parameters provided - will validate consistency", style="yellow")
    
    try:
        result = convert_survival_parameters(
            median_survival=median,
            hazard_rate=hazard,
            survival_fraction=survival_fraction,
            event_rate=event_rate,
            time_point=time_point
        )
        
        _display_conversion_results(result, time_unit, output_format)
        
    except ValueError as e:
        console.print(f"Error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def hazard_ratio(
    hazard_ratio: float = typer.Option(..., help='Hazard ratio (treatment vs control)'),
    control_median: Optional[float] = typer.Option(None, help='Control group median survival'),
    treatment_median: Optional[float] = typer.Option(None, help='Treatment group median survival'),
    control_hazard: Optional[float] = typer.Option(None, help='Control group hazard rate'),
    treatment_hazard: Optional[float] = typer.Option(None, help='Treatment group hazard rate'),
    time_point: float = typer.Option(12.0, help='Time point for fractions'),
    time_unit: str = typer.Option('months', help='Time unit'),
    output_format: str = typer.Option('table', help='Output format (table/json/compact)')
):
    """
    Convert hazard ratio scenario to complete parameter sets.
    
    Provide hazard ratio and any one parameter from either group.
    
    Examples:
        designpower survival-converter hazard-ratio --hazard-ratio 0.7 --control-median 12
        designpower survival-converter hazard-ratio --hazard-ratio 0.67 --treatment-median 18
    """
    try:
        result = convert_hazard_ratio_scenario(
            hazard_ratio=hazard_ratio,
            control_median=control_median,
            treatment_median=treatment_median,
            control_hazard=control_hazard,
            treatment_hazard=treatment_hazard,
            time_point=time_point
        )
        
        _display_hazard_ratio_results(result, time_unit, output_format)
        
    except ValueError as e:
        console.print(f"Error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def units(
    value: float = typer.Option(..., help='Value to convert'),
    parameter: str = typer.Option(..., help='Parameter type (median/hazard)'),
    from_unit: str = typer.Option(..., help='Original time unit (days/weeks/months/years)'),
    to_unit: str = typer.Option(..., help='Target time unit (days/weeks/months/years)'),
    time_point: float = typer.Option(12.0, help='Time point for fractions'),
    time_point_unit: str = typer.Option('months', help='Time point unit')
):
    """
    Convert survival parameters between time units.
    
    Examples:
        designpower survival-converter units --value 1 --parameter median --from-unit years --to-unit months
        designpower survival-converter units --value 0.058 --parameter hazard --from-unit months --to-unit years
    """
    try:
        result = convert_survival_parameters_with_units(
            value=value,
            parameter_type=parameter,
            from_unit=from_unit,
            to_unit=to_unit,
            time_point=time_point,
            time_point_unit=time_point_unit
        )
        
        console.print(f"\n=== Unit Conversion: {value} ({parameter}) from {from_unit} to {to_unit} ===")
        _display_conversion_results(result, to_unit, 'table')
        
    except ValueError as e:
        console.print(f"Error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def examples():
    """Show practical examples of survival parameter conversion."""
    
    examples_text = """
[bold blue]=== Survival Parameter Converter Examples ===[/bold blue]

[bold]1. CONVERTING FROM MEDIAN SURVIVAL:[/bold]
   [cyan]designpower survival-converter convert --median 12[/cyan]
   Use case: You know median survival from prior studies
   
[bold]2. CONVERTING FROM SURVIVAL FRACTION:[/bold]
   [cyan]designpower survival-converter convert --survival-fraction 0.7 --time-point 24[/cyan]
   Use case: You know 70% survive to 24 months from literature
   
[bold]3. HAZARD RATIO SCENARIO:[/bold]
   [cyan]designpower survival-converter hazard-ratio --hazard-ratio 0.67 --control-median 12[/cyan]
   Use case: Planning trial with HR=0.67, control median=12 months
   
[bold]4. UNIT CONVERSIONS:[/bold]
   [cyan]designpower survival-converter units --value 1 --parameter median --from-unit years --to-unit months[/cyan]
   Use case: Converting between time units for consistency

[bold green]=== Common Clinical Trial Scenarios ===[/bold green]

[bold]Cancer Trial Planning:[/bold]
• Have: Control median = 12 months, target HR = 0.7
• Need: Treatment median, event rates, hazard rates
• Command: [cyan]designpower survival-converter hazard-ratio --hazard-ratio 0.7 --control-median 12[/cyan]

[bold]Literature Meta-Analysis:[/bold]
• Have: Study reports "60% 5-year survival"
• Need: Median survival, hazard rate for power calculation  
• Command: [cyan]designpower survival-converter convert --survival-fraction 0.6 --time-point 60[/cyan]

[bold]Protocol Development:[/bold]
• Have: Historical event rate of 40% at 2 years
• Need: Median survival for sample size calculation
• Command: [cyan]designpower survival-converter convert --event-rate 0.4 --time-point 24[/cyan]
"""
    
    console.print(examples_text)


def _display_conversion_results(result: dict, time_unit: str, output_format: str):
    """Display conversion results in specified format."""
    
    if output_format == 'json':
        console.print(json.dumps(result, indent=2))
        
    elif output_format == 'compact':
        console.print(f"Median: {result['median_survival']:.3f} {time_unit}, "
                     f"Hazard: {result['hazard_rate']:.4f}/{time_unit}, "
                     f"Survival@{result['time_point']}: {result['survival_fraction']:.1%}, "
                     f"Events@{result['time_point']}: {result['event_rate']:.1%}")
        
    else:  # table format
        console.print(f"\n[bold blue]=== Survival Parameter Conversion Results ===[/bold blue]")
        console.print(f"Time unit: {time_unit}")
        console.print(f"Time point for fractions: {result['time_point']:.1f} {time_unit}")
        console.print()
        console.print(f"[bold]Median survival:[/bold]       {result['median_survival']:.3f} {time_unit}")
        console.print(f"[bold]Instantaneous hazard:[/bold]  {result['hazard_rate']:.4f} per {time_unit}")
        console.print(f"[bold]Survival fraction:[/bold]     {result['survival_fraction']:.1%} at {result['time_point']:.1f} {time_unit}")
        console.print(f"[bold]Event rate:[/bold]           {result['event_rate']:.1%} by {result['time_point']:.1f} {time_unit}")


def _display_hazard_ratio_results(result: dict, time_unit: str, output_format: str):
    """Display hazard ratio scenario results in specified format."""
    
    if output_format == 'json':
        console.print(json.dumps(result, indent=2))
        
    elif output_format == 'compact':
        control = result['control']
        treatment = result['treatment']
        console.print(f"HR: {result['hazard_ratio']:.3f}, "
                     f"Control median: {control['median_survival']:.2f}, "
                     f"Treatment median: {treatment['median_survival']:.2f}")
        
    else:  # table format
        control = result['control']
        treatment = result['treatment']
        
        console.print(f"\n[bold blue]=== Hazard Ratio Scenario Results ===[/bold blue]")
        console.print(f"[bold]Hazard Ratio:[/bold] {result['hazard_ratio']:.3f} (treatment vs control)")
        console.print(f"Time unit: {time_unit}")
        console.print(f"Time point: {control['time_point']:.1f} {time_unit}")
        console.print()
        
        # Table format
        console.print("[bold]Parameter                    Control      Treatment[/bold]")
        console.print("-" * 50)
        console.print(f"Median survival:            {control['median_survival']:8.2f}     {treatment['median_survival']:8.2f} {time_unit}")
        console.print(f"Hazard rate:                {control['hazard_rate']:8.4f}     {treatment['hazard_rate']:8.4f} per {time_unit}")
        console.print(f"Survival fraction:          {control['survival_fraction']:8.1%}     {treatment['survival_fraction']:8.1%} at {control['time_point']:.0f} {time_unit}")
        console.print(f"Event rate:                 {control['event_rate']:8.1%}     {treatment['event_rate']:8.1%} by {control['time_point']:.0f} {time_unit}")
        
        # Verify hazard ratio
        calculated_hr = result.get('calculated_hazard_ratio', 'N/A')
        if isinstance(calculated_hr, float):
            console.print(f"\n[dim]Verification: Calculated HR = {calculated_hr:.4f}[/dim]")


if __name__ == '__main__':
    app()