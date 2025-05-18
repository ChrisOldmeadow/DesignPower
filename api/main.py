"""
FastAPI backend for sample size calculator.

This module provides REST API endpoints for sample size calculation,
power analysis, and simulation-based estimation.
"""
from typing import Dict, Any, List, Optional, Union
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
from core.utils import generate_code_snippet, generate_plain_language_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sample Size Calculator API",
    description="API for sample size, power calculation, and simulation-based estimation",
    version="0.1.0",
)


# Input models for each endpoint
class DifferenceInMeansInput(BaseModel):
    delta: float = Field(..., description="Minimum detectable effect (difference between means)")
    std_dev: float = Field(..., description="Pooled standard deviation of the outcome")
    power: float = Field(0.8, description="Desired statistical power (1 - beta)")
    alpha: float = Field(0.05, description="Significance level")
    allocation_ratio: float = Field(1.0, description="Ratio of sample sizes (n2/n1)")


class DifferenceInMeansPowerInput(BaseModel):
    n1: int = Field(..., description="Sample size for group 1")
    n2: int = Field(..., description="Sample size for group 2")
    delta: float = Field(..., description="Minimum detectable effect (difference between means)")
    std_dev: float = Field(..., description="Pooled standard deviation of the outcome")
    alpha: float = Field(0.05, description="Significance level")


class BinaryClusterRCTInput(BaseModel):
    n_clusters: int = Field(..., description="Number of clusters per arm")
    cluster_size: int = Field(..., description="Average number of individuals per cluster")
    icc: float = Field(..., description="Intracluster correlation coefficient")
    p1: float = Field(..., description="Proportion in control group")
    p2: float = Field(..., description="Proportion in intervention group")
    alpha: float = Field(0.05, description="Significance level")


class BinaryClusterRCTSampleSizeInput(BaseModel):
    p1: float = Field(..., description="Proportion in control group")
    p2: float = Field(..., description="Proportion in intervention group")
    icc: float = Field(..., description="Intracluster correlation coefficient")
    cluster_size: int = Field(..., description="Average number of individuals per cluster")
    power: float = Field(0.8, description="Desired statistical power (1 - beta)")
    alpha: float = Field(0.05, description="Significance level")


class BinaryClusterRCTMDEInput(BaseModel):
    n_clusters: int = Field(..., description="Number of clusters per arm")
    cluster_size: int = Field(..., description="Average number of individuals per cluster")
    icc: float = Field(..., description="Intracluster correlation coefficient")
    p1: float = Field(..., description="Proportion in control group")
    power: float = Field(0.8, description="Desired statistical power (1 - beta)")
    alpha: float = Field(0.05, description="Significance level")


class ParallelRCTSimulationInput(BaseModel):
    n1: int = Field(..., description="Sample size for group 1")
    n2: int = Field(..., description="Sample size for group 2")
    mean1: float = Field(..., description="Mean outcome in group 1")
    mean2: float = Field(..., description="Mean outcome in group 2")
    std_dev: float = Field(..., description="Standard deviation of outcome")
    nsim: int = Field(1000, description="Number of simulations")
    alpha: float = Field(0.05, description="Significance level")


class ClusterRCTSimulationInput(BaseModel):
    n_clusters: int = Field(..., description="Number of clusters per arm")
    cluster_size: int = Field(..., description="Number of individuals per cluster")
    icc: float = Field(..., description="Intracluster correlation coefficient")
    mean1: float = Field(..., description="Mean outcome in control arm")
    mean2: float = Field(..., description="Mean outcome in intervention arm")
    std_dev: float = Field(..., description="Total standard deviation of outcome")
    nsim: int = Field(1000, description="Number of simulations")
    alpha: float = Field(0.05, description="Significance level")


class SteppedWedgeSimulationInput(BaseModel):
    clusters: int = Field(..., description="Number of clusters")
    steps: int = Field(..., description="Number of time steps (including baseline)")
    individuals_per_cluster: int = Field(..., description="Number of individuals per cluster per time step")
    icc: float = Field(..., description="Intracluster correlation coefficient")
    treatment_effect: float = Field(..., description="Effect size of the intervention")
    std_dev: float = Field(..., description="Total standard deviation of outcome")
    nsim: int = Field(1000, description="Number of simulations")
    alpha: float = Field(0.05, description="Significance level")


class BinaryClusterRCTSimulationInput(BaseModel):
    n_clusters: int = Field(..., description="Number of clusters per arm")
    cluster_size: int = Field(..., description="Number of individuals per cluster")
    icc: float = Field(..., description="Intracluster correlation coefficient")
    p1: float = Field(..., description="Proportion in control arm")
    p2: float = Field(..., description="Proportion in intervention arm")
    nsim: int = Field(1000, description="Number of simulations")
    alpha: float = Field(0.05, description="Significance level")


class ResultResponse(BaseModel):
    result: Dict[str, Any] = Field(..., description="Calculation result")
    code_snippet: str = Field(..., description="Reproducible code snippet")
    summary: str = Field(..., description="Plain language summary")


# Routes
@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Sample Size Calculator API",
        "version": "0.1.0",
        "endpoints": [
            "/calculate/sample-size",
            "/calculate/power",
            "/calculate/simulation"
        ]
    }


@app.post("/calculate/sample-size/difference-in-means", response_model=ResultResponse)
async def calc_sample_size_difference_in_means(input_data: DifferenceInMeansInput):
    """Calculate sample size for difference in means."""
    try:
        result = sample_size_difference_in_means(
            delta=input_data.delta,
            std_dev=input_data.std_dev,
            power=input_data.power,
            alpha=input_data.alpha,
            allocation_ratio=input_data.allocation_ratio
        )
        
        code_snippet = generate_code_snippet("sample_size_difference_in_means", result["parameters"])
        summary = generate_plain_language_summary("sample_size_difference_in_means", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in sample size calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/power/difference-in-means", response_model=ResultResponse)
async def calc_power_difference_in_means(input_data: DifferenceInMeansPowerInput):
    """Calculate power for difference in means."""
    try:
        result = power_difference_in_means(
            n1=input_data.n1,
            n2=input_data.n2,
            delta=input_data.delta,
            std_dev=input_data.std_dev,
            alpha=input_data.alpha
        )
        
        code_snippet = generate_code_snippet("power_difference_in_means", result["parameters"])
        summary = generate_plain_language_summary("power_difference_in_means", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in power calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/power/binary-cluster-rct", response_model=ResultResponse)
async def calc_power_binary_cluster_rct(input_data: BinaryClusterRCTInput):
    """Calculate power for binary outcome in cluster RCT."""
    try:
        result = power_binary_cluster_rct(
            n_clusters=input_data.n_clusters,
            cluster_size=input_data.cluster_size,
            icc=input_data.icc,
            p1=input_data.p1,
            p2=input_data.p2,
            alpha=input_data.alpha
        )
        
        code_snippet = generate_code_snippet("power_binary_cluster_rct", result["parameters"])
        summary = generate_plain_language_summary("power_binary_cluster_rct", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in power calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/sample-size/binary-cluster-rct", response_model=ResultResponse)
async def calc_sample_size_binary_cluster_rct(input_data: BinaryClusterRCTSampleSizeInput):
    """Calculate sample size for binary outcome in cluster RCT."""
    try:
        result = sample_size_binary_cluster_rct(
            p1=input_data.p1,
            p2=input_data.p2,
            icc=input_data.icc,
            cluster_size=input_data.cluster_size,
            power=input_data.power,
            alpha=input_data.alpha
        )
        
        code_snippet = generate_code_snippet("sample_size_binary_cluster_rct", result["parameters"])
        summary = generate_plain_language_summary("sample_size_binary_cluster_rct", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in sample size calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/mde/binary-cluster-rct", response_model=ResultResponse)
async def calc_mde_binary_cluster_rct(input_data: BinaryClusterRCTMDEInput):
    """Calculate minimum detectable effect for binary outcome in cluster RCT."""
    try:
        result = min_detectable_effect_binary_cluster_rct(
            n_clusters=input_data.n_clusters,
            cluster_size=input_data.cluster_size,
            icc=input_data.icc,
            p1=input_data.p1,
            power=input_data.power,
            alpha=input_data.alpha
        )
        
        code_snippet = generate_code_snippet("min_detectable_effect_binary_cluster_rct", result["parameters"])
        summary = generate_plain_language_summary("min_detectable_effect_binary_cluster_rct", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in MDE calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/simulation/parallel-rct", response_model=ResultResponse)
async def calc_simulation_parallel_rct(input_data: ParallelRCTSimulationInput):
    """Perform simulation for parallel RCT with continuous outcome."""
    try:
        result = simulate_parallel_rct(
            n1=input_data.n1,
            n2=input_data.n2,
            mean1=input_data.mean1,
            mean2=input_data.mean2,
            std_dev=input_data.std_dev,
            nsim=input_data.nsim,
            alpha=input_data.alpha
        )
        
        code_snippet = generate_code_snippet("simulate_parallel_rct", result["parameters"])
        summary = generate_plain_language_summary("simulate_parallel_rct", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/simulation/cluster-rct", response_model=ResultResponse)
async def calc_simulation_cluster_rct(input_data: ClusterRCTSimulationInput):
    """Perform simulation for cluster RCT with continuous outcome."""
    try:
        result = simulate_cluster_rct(
            n_clusters=input_data.n_clusters,
            cluster_size=input_data.cluster_size,
            icc=input_data.icc,
            mean1=input_data.mean1,
            mean2=input_data.mean2,
            std_dev=input_data.std_dev,
            nsim=input_data.nsim,
            alpha=input_data.alpha
        )
        
        code_snippet = generate_code_snippet("simulate_cluster_rct", result["parameters"])
        summary = generate_plain_language_summary("simulate_cluster_rct", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/simulation/stepped-wedge", response_model=ResultResponse)
async def calc_simulation_stepped_wedge(input_data: SteppedWedgeSimulationInput):
    """Perform simulation for stepped wedge design."""
    try:
        result = simulate_stepped_wedge(
            clusters=input_data.clusters,
            steps=input_data.steps,
            individuals_per_cluster=input_data.individuals_per_cluster,
            icc=input_data.icc,
            treatment_effect=input_data.treatment_effect,
            std_dev=input_data.std_dev,
            nsim=input_data.nsim,
            alpha=input_data.alpha
        )
        
        code_snippet = generate_code_snippet("simulate_stepped_wedge", result["parameters"])
        summary = generate_plain_language_summary("simulate_stepped_wedge", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate/simulation/binary-cluster-rct", response_model=ResultResponse)
async def calc_simulation_binary_cluster_rct(input_data: BinaryClusterRCTSimulationInput):
    """Perform simulation for cluster RCT with binary outcome."""
    try:
        result = simulate_binary_cluster_rct(
            n_clusters=input_data.n_clusters,
            cluster_size=input_data.cluster_size,
            icc=input_data.icc,
            p1=input_data.p1,
            p2=input_data.p2,
            nsim=input_data.nsim,
            alpha=input_data.alpha
        )
        
        code_snippet = generate_code_snippet("simulate_binary_cluster_rct", result["parameters"])
        summary = generate_plain_language_summary("simulate_binary_cluster_rct", result)
        
        return {
            "result": result,
            "code_snippet": code_snippet,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
