"""
Simplified main CLI for DesignPower.

This replaces the oversized cli.py with a more modular structure.
"""
import typer
from cli.shared import console
from cli.commands import parallel, cluster, single_arm, survival, utilities

app = typer.Typer(help="DesignPower: Sample Size and Power Calculation Tool")

# Add command groups
app.add_typer(parallel.app, name="parallel", help="Parallel group RCT calculations")
app.add_typer(cluster.app, name="cluster", help="Cluster randomized trial calculations") 
app.add_typer(single_arm.app, name="single-arm", help="Single-arm trial calculations")
app.add_typer(survival.app, name="survival", help="Survival analysis calculations")
app.add_typer(utilities.app, name="utils", help="Utility commands")

@app.callback()
def main():
    """
    DesignPower CLI for statistical power analysis and sample size calculations.
    
    Examples:
        designpower parallel sample-size --outcome continuous --delta 0.5 --std-dev 1.0
        designpower cluster sample-size --outcome binary --p1 0.3 --p2 0.5 --cluster-size 20 --icc 0.05
        designpower single-arm sample-size --design ahern --p 0.3 --p0 0.1
    """
    pass

if __name__ == "__main__":
    app()