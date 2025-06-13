"""
CLI utility commands.

This module contains utility commands for script generation,
examples, and other helper functions.
"""
import typer
import json
from pathlib import Path

from cli.shared import console

app = typer.Typer(help="Utility Commands")


@app.command()
def julia():
    """
    Display Julia integration example.
    """
    console.print("""
[bold green]Julia Integration Example[/bold green]

DesignPower includes Julia backend support for specialized calculations.

Example Julia script for stepped wedge design:
""")
    
    julia_code = '''
using Statistics, Random, Distributions

function stepped_wedge_power(clusters, steps, individuals_per_cluster, 
                           icc, treatment_effect, std_dev, alpha=0.05)
    # Julia implementation for stepped wedge power calculation
    # This is a placeholder - actual implementation would go here
    return 0.8  # Example return value
end

# Example usage
power = stepped_wedge_power(6, 4, 20, 0.05, 0.5, 1.0)
println("Power: ", power)
'''
    
    console.print(f"[dim]{julia_code}[/dim]")
    console.print("\n[bold blue]Note:[/bold blue] This is a demonstration. Actual Julia backend integration is available for stepped wedge designs.")


@app.command()
def examples():
    """
    Display common usage examples.
    """
    console.print("""
[bold green]DesignPower CLI Examples[/bold green]

[bold]Parallel Group RCT:[/bold]
  designpower parallel sample-size --outcome continuous --delta 0.5 --std-dev 1.0 --power 0.8
  designpower parallel sample-size --outcome binary --p1 0.3 --p2 0.5 --power 0.9

[bold]Cluster Randomized Trial:[/bold]
  designpower cluster sample-size --outcome binary --p1 0.3 --p2 0.5 --cluster-size 20 --icc 0.05
  designpower cluster sample-size --outcome continuous --delta 0.5 --std-dev 1.0 --cluster-size 15 --icc 0.03

[bold]Single-Arm Trial:[/bold]
  designpower single-arm sample-size --design standard --outcome binary --p 0.3 --p0 0.1
  designpower single-arm sample-size --design ahern --outcome binary --p 0.4 --p0 0.2
  designpower single-arm sample-size --design simons --outcome binary --p 0.3 --p0 0.1 --simon-type optimal

[bold]Survival Analysis:[/bold]
  designpower survival sample-size --design parallel --median-control 12 --median-treatment 18
  designpower survival sample-size --design single-arm --median-null 12 --median-treatment 18

[bold]Generate Reproducible Scripts:[/bold]
  designpower parallel sample-size --outcome continuous --delta 0.5 --std-dev 1.0 --script --script-file my_analysis.py
  designpower cluster sample-size --outcome binary --p1 0.3 --p2 0.5 --cluster-size 20 --icc 0.05 --script

[bold]Get Help:[/bold]
  designpower --help
  designpower parallel --help
  designpower cluster sample-size --help
""")


@app.command()
def validate():
    """
    Run validation tests against known benchmarks.
    """
    console.print("[bold blue]Running validation tests...[/bold blue]")
    
    try:
        from tests.validation.run_validation import main as run_validation
        run_validation()
        console.print("[bold green]✓ Validation tests completed[/bold green]")
    except ImportError:
        console.print("[bold yellow]Warning:[/bold yellow] Validation module not available")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Validation failed: {str(e)}")


@app.command()
def info():
    """
    Display information about DesignPower.
    """
    console.print("""
[bold green]DesignPower[/bold green] - Statistical Power Analysis Tool

[bold]Version:[/bold] 1.0.0
[bold]Description:[/bold] Comprehensive statistical power analysis application for clinical trial design

[bold]Supported Designs:[/bold]
• Parallel group RCTs (continuous, binary, survival outcomes)
• Cluster randomized trials (continuous, binary outcomes)  
• Single-arm trials (continuous, binary, survival outcomes)
• Non-inferiority tests
• Permutation tests
• A'Hern and Simon's two-stage designs

[bold]Key Features:[/bold]
• Algorithm transparency with source code in generated scripts
• Validation against statistical literature benchmarks  
• Both analytical and simulation-based methods
• Rich CLI interface with comprehensive help
• Reproducible Python script generation

[bold]Documentation:[/bold]
See CLAUDE.md for project guidelines and implementation details.
""")