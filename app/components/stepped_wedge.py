"""
Component module for Stepped Wedge Cluster RCT designs.

This module provides UI rendering functions and calculation functions for
Stepped Wedge designs with continuous and binary outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import graphviz

# Import design-specific modules
from core.designs.stepped_wedge.simulation import simulate_continuous, simulate_binary
from core.designs.stepped_wedge.analytical import (
    hussey_hughes_power_continuous, hussey_hughes_sample_size_continuous,
    hussey_hughes_power_binary, hussey_hughes_sample_size_binary
)


def create_stepped_wedge_visualization(clusters, steps):
    """
    Create an adaptive visualization of the stepped wedge design.
    Uses heatmap for small studies, line plot for large studies.
    
    Args:
        clusters: Number of clusters
        steps: Number of time steps
        
    Returns:
        matplotlib figure
    """
    # Create the design matrix
    design_matrix = np.zeros((clusters, steps))
    
    # Create a proper stepped wedge pattern (staircase)
    # Distribute clusters evenly across intervention steps (excluding baseline step 0)
    clusters_per_step = clusters // (steps - 1)
    remaining_clusters = clusters % (steps - 1)
    
    cluster_step_assignment = []
    current_cluster = 0
    
    for step in range(1, steps):  # Start from step 1 (skip baseline)
        # Number of clusters for this step
        clusters_for_this_step = clusters_per_step
        if step <= remaining_clusters:  # Distribute remaining clusters to early steps
            clusters_for_this_step += 1
        
        # Assign clusters to this intervention step
        for _ in range(clusters_for_this_step):
            if current_cluster < clusters:
                cluster_step_assignment.append(step)
                current_cluster += 1
    
    cluster_step_assignment = np.array(cluster_step_assignment)
    
    # Fill the design matrix
    for c in range(clusters):
        step_of_intervention = cluster_step_assignment[c]
        for t in range(steps):
            if t >= step_of_intervention:
                design_matrix[c, t] = 1  # Intervention period
            else:
                design_matrix[c, t] = 0  # Control period
    
    # Calculate design metrics
    total_periods = clusters * steps
    control_periods = np.sum(design_matrix == 0)
    intervention_periods = np.sum(design_matrix == 1)
    
    # Adaptive visualization based on study size
    if clusters > 20 or clusters * steps > 300:
        # Use line plot for large studies
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, steps * 0.8), 8), 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Top plot: Cumulative number of clusters under intervention
        intervention_start_times = []
        cumulative_intervention = np.zeros(steps)
        
        # Calculate when each cluster starts intervention and cumulative counts
        for step in range(steps):
            cumulative_intervention[step] = np.sum(design_matrix[:, step])
        
        # Create stepped line plot
        x_steps = np.arange(steps)
        ax1.step(x_steps, cumulative_intervention, where='mid', linewidth=3, 
                color='#2E86AB', label='Clusters under intervention')
        ax1.fill_between(x_steps, 0, cumulative_intervention, step='mid', 
                        alpha=0.3, color='#2E86AB')
        
        # Add markers at transition points
        transition_points = []
        for step in range(1, steps):
            n_new_clusters = cumulative_intervention[step] - cumulative_intervention[step-1]
            if n_new_clusters > 0:
                transition_points.append((step, cumulative_intervention[step], n_new_clusters))
        
        for step, total, new in transition_points:
            ax1.annotate(f'+{int(new)} clusters', 
                        xy=(step, total), xytext=(step, total + clusters*0.1),
                        ha='center', va='bottom', fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))
        
        ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Clusters\\nUnder Intervention', fontsize=12, fontweight='bold')
        ax1.set_title(f'Stepped Wedge Design: {clusters} Clusters, {steps} Steps', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(range(steps))
        ax1.set_xticklabels([f'Step {i+1}' for i in range(steps)])
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, clusters * 1.1)
        
        # Bottom plot: Design efficiency over time
        efficiency = cumulative_intervention / clusters
        ax2.plot(x_steps, efficiency, 'o-', color='#E67E22', linewidth=2, markersize=6)
        ax2.fill_between(x_steps, 0, efficiency, alpha=0.2, color='#E67E22')
        ax2.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Proportion Under\\nIntervention', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(steps))
        ax2.set_xticklabels([f'Step {i+1}' for i in range(steps)])
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # Add design summary text
        info_text = f"""Design Summary:
• Total Cluster-Periods: {total_periods:,}
• Control Periods: {control_periods:,} ({control_periods/total_periods:.1%})
• Intervention Periods: {intervention_periods:,} ({intervention_periods/total_periods:.1%})
• Design Efficiency: {intervention_periods/total_periods:.1%}"""
        
        fig.text(0.98, 0.02, info_text, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
    else:
        # Use heatmap for smaller studies (≤20 clusters)
        fig, ax = plt.subplots(figsize=(max(8, steps * 0.8), max(6, clusters * 0.4)))
        
        # Create custom colormap
        colors = ['#E8F4FD', '#2E86AB']  # Light blue for control, dark blue for intervention
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        # Create the heatmap
        im = ax.imshow(design_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(steps))
        ax.set_xticklabels([f'Step {i+1}' for i in range(steps)])
        ax.set_yticks(range(clusters))
        ax.set_yticklabels([f'Cluster {i+1}' for i in range(clusters)])
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, steps, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, clusters, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        
        # Labels and title
        ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clusters', fontsize=12, fontweight='bold')
        ax.set_title('Stepped Wedge Design Layout', fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations only if not too many cells
        if clusters * steps <= 100:
            for i in range(clusters):
                for j in range(steps):
                    text = 'I' if design_matrix[i, j] == 1 else 'C'
                    color = 'white' if design_matrix[i, j] == 1 else 'black'
                    ax.text(j, i, text, ha='center', va='center', 
                           color=color, fontweight='bold', fontsize=10)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], label='Control Period (C)'),
            Patch(facecolor=colors[1], label='Intervention Period (I)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Add design information text
        info_text = f"""Design Summary:
• Total Cluster-Periods: {total_periods:,}
• Control Periods: {control_periods:,}
• Intervention Periods: {intervention_periods:,}
• Baseline (Step 1): All clusters in control
• Implementation: Staggered across {steps-1} steps"""
        
        ax.text(1.02, 0.5, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def display_stepped_wedge_design(clusters, steps):
    """
    Display the stepped wedge design visualization in Streamlit.
    
    Args:
        clusters: Number of clusters
        steps: Number of time steps
    """
    with st.expander("📊 Stepped Wedge Design Visualization", expanded=False):
        # Adaptive description based on visualization type
        if clusters > 20 or clusters * steps > 300:
            st.write("**Line Plot View** (for large studies): Shows cumulative intervention rollout and design efficiency over time")
        else:
            st.write("**Heatmap View** (for smaller studies): Shows when each cluster receives the intervention")
        
        try:
            # Create and display the plot with unique key for proper refresh
            fig = create_stepped_wedge_visualization(clusters, steps)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)  # Close to free memory
        except Exception as e:
            st.warning(f"Unable to display visualization. Error: {str(e)}")
            # Provide alternative text-based representation
            st.write("**Design Structure:**")
            st.write(f"- {clusters} clusters")
            st.write(f"- {steps} time steps")
            st.write(f"- Clusters transition from control to intervention in a stepped pattern")
        
        # Add adaptive explanatory text
        if clusters > 20 or clusters * steps > 300:
            st.write("""
            **How to read this chart:**
            - **Top panel**: Shows cumulative number of clusters receiving intervention over time
            - **Bottom panel**: Shows proportion of clusters under intervention (design efficiency)
            - **Red arrows**: Indicate when new clusters start the intervention
            - All clusters start in control, then switch to intervention at staggered time points
            - Once a cluster switches to intervention, it stays in intervention (no crossover back)
            """)
        else:
            st.write("""
            **How to read this chart:**
            - Each row represents a cluster
            - Each column represents a time step
            - **C** = Control period (light blue)
            - **I** = Intervention period (dark blue)
            - All clusters start in control, then switch to intervention at different steps
            - Once a cluster switches to intervention, it stays in intervention
            """)
        
        # Calculate and display design efficiency metrics using the proper stepped wedge pattern
        total_periods = clusters * steps
        
        # Calculate actual control and intervention periods based on staircase pattern
        control_periods = 0
        intervention_periods = 0
        
        # Recreate the assignment logic to get accurate counts
        clusters_per_step = clusters // (steps - 1)
        remaining_clusters = clusters % (steps - 1)
        
        current_cluster = 0
        for step in range(1, steps):  # Start from step 1 (skip baseline)
            clusters_for_this_step = clusters_per_step
            if step <= remaining_clusters:
                clusters_for_this_step += 1
            
            # For each cluster in this step, count periods
            for _ in range(clusters_for_this_step):
                if current_cluster < clusters:
                    control_periods += step  # Periods before intervention
                    intervention_periods += (steps - step)  # Periods during intervention
                    current_cluster += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Control Periods", f"{control_periods}")
        with col2:
            st.metric("Intervention Periods", f"{intervention_periods}")
        with col3:
            efficiency = intervention_periods / total_periods
            st.metric("Design Efficiency", f"{efficiency:.1%}")
            
        st.caption("Design Efficiency = Proportion of cluster-periods under intervention")


def render_stepped_wedge_continuous(calc_type, hypothesis_type):
    """
    Render the UI for Stepped Wedge design with continuous outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Stepped Wedge Cluster RCT with Continuous Outcome ({calc_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Study Design Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["clusters"] = st.number_input("Number of Clusters", 
                                               value=12, step=1, min_value=3, max_value=500, 
                                               help="Total number of clusters in the study",
                                               key="sw_clusters_continuous")
            
            params["steps"] = st.number_input("Number of Time Steps", 
                                            value=4, step=1, min_value=2, max_value=20,
                                            help="Total time steps including baseline (min 2)",
                                            key="sw_steps_continuous")
            
            params["individuals_per_cluster"] = st.number_input("Individuals per Cluster per Step", 
                                                              value=25, step=1, min_value=1, max_value=10000,
                                                              help="Number of individuals in each cluster at each time step",
                                                              key="sw_indiv_continuous")
            
        with col2:
            params["icc"] = st.slider("Intracluster Correlation (ICC)", 
                                    min_value=0.0, max_value=0.3, value=0.05, step=0.01,
                                    help="Correlation between individuals within the same cluster",
                                    key="sw_icc_continuous")
            
            params["treatment_effect"] = st.number_input("Treatment Effect", 
                                                       value=0.5, step=0.1,
                                                       help="Expected difference in means due to intervention",
                                                       key="sw_effect_continuous")
            
            params["std_dev"] = st.number_input("Standard Deviation", 
                                              value=2.0, step=0.1, min_value=0.1,
                                              help="Standard deviation of the outcome",
                                              key="sw_sd_continuous")

    # Statistical parameters
    with st.container():
        st.subheader("Statistical Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params["alpha"] = st.slider("Significance Level (α)", 
                                      min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                      key="sw_alpha_continuous")
        
        with col2:
            params["nsim"] = st.selectbox("Number of Simulations", 
                                        options=[100, 500, 1000, 2000, 5000], 
                                        index=2,
                                        help="More simulations = more accurate results but slower computation",
                                        key="sw_nsim_continuous")

    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        st.subheader("Methodological Approach")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params["method"] = st.radio("Analysis Method", 
                                      options=["Simulation", "Hussey & Hughes Analytical"],
                                      help="Simulation uses Monte Carlo; Analytical uses Hussey & Hughes (2007) closed-form formulas",
                                      key="sw_method_continuous")
        
        with col2:
            if params["method"] == "Hussey & Hughes Analytical":
                st.info("Using analytical formulas from Hussey & Hughes (2007)")
            else:
                st.info(f"Using {params['nsim']} Monte Carlo simulations")
        
        st.subheader("Correlation Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Within-cluster, within-period correlation (ICC):**")
            st.write(f"ICC = {params['icc']:.3f}")
            st.caption("Correlation between individuals in same cluster at same time")
        
        with col2:
            params["cluster_autocorr"] = st.slider("Cluster Autocorrelation (CAC)", 
                                                  min_value=0.0, max_value=0.9, value=0.0, step=0.01,
                                                  help="Within-cluster, between-period correlation - correlation within clusters over time",
                                                  key="sw_cac_continuous")
            st.caption("Correlation within clusters across time periods")
            
        if params["cluster_autocorr"] > 0:
            st.info(f"""
            **Advanced correlation structure enabled:**
            - ICC (within-period): {params['icc']:.3f}
            - CAC (between-period): {params['cluster_autocorr']:.3f}
            """)
        else:
            st.info("Using standard correlation structure (ICC only)")

    # Display study design summary
    with st.expander("Study Design Summary", expanded=False):
        total_n = params["clusters"] * params["steps"] * params["individuals_per_cluster"]
        control_periods = params["clusters"] * 1  # Baseline only
        intervention_periods = params["clusters"] * (params["steps"] - 1)
        
        st.write(f"**Total Sample Size:** {total_n:,} individuals")
        st.write(f"**Total Cluster-Periods:** {params['clusters'] * params['steps']}")
        st.write(f"**Control Periods:** {control_periods}")
        st.write(f"**Intervention Periods:** {intervention_periods}")
        st.write(f"**Design Effect (approx):** {1 + (params['individuals_per_cluster'] - 1) * params['icc']:.2f}")

    # Add button to preview design layout (safe visualization)
    if st.button("🔍 Preview Design Layout", help="Generate a visual preview of the stepped wedge design"):
        try:
            display_stepped_wedge_design(params["clusters"], params["steps"])
        except Exception as e:
            st.error(f"Could not generate design preview: {str(e)}")

    return params


def render_stepped_wedge_binary(calc_type, hypothesis_type):
    """
    Render the UI for Stepped Wedge design with binary outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Stepped Wedge Cluster RCT with Binary Outcome ({calc_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Study Design Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["clusters"] = st.number_input("Number of Clusters", 
                                               value=12, step=1, min_value=3, max_value=500, 
                                               help="Total number of clusters in the study",
                                               key="sw_clusters_binary")
            
            params["steps"] = st.number_input("Number of Time Steps", 
                                            value=4, step=1, min_value=2, max_value=20,
                                            help="Total time steps including baseline (min 2)",
                                            key="sw_steps_binary")
            
            params["individuals_per_cluster"] = st.number_input("Individuals per Cluster per Step", 
                                                              value=25, step=1, min_value=1, max_value=10000,
                                                              help="Number of individuals in each cluster at each time step",
                                                              key="sw_indiv_binary")
            
        with col2:
            params["icc"] = st.slider("Intracluster Correlation (ICC)", 
                                    min_value=0.0, max_value=0.3, value=0.05, step=0.01,
                                    help="Correlation between individuals within the same cluster",
                                    key="sw_icc_binary")
            
            params["p_control"] = st.slider("Control Proportion", 
                                          min_value=0.01, max_value=0.99, value=0.30, step=0.01,
                                          help="Expected proportion of events in control condition",
                                          key="sw_p_control")
            
            params["p_intervention"] = st.slider("Intervention Proportion", 
                                                min_value=0.01, max_value=0.99, value=0.45, step=0.01,
                                                help="Expected proportion of events under intervention",
                                                key="sw_p_intervention")

    # Statistical parameters
    with st.container():
        st.subheader("Statistical Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params["alpha"] = st.slider("Significance Level (α)", 
                                      min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                      key="sw_alpha_binary")
        
        with col2:
            params["nsim"] = st.selectbox("Number of Simulations", 
                                        options=[100, 500, 1000, 2000, 5000], 
                                        index=2,
                                        help="More simulations = more accurate results but slower computation",
                                        key="sw_nsim_binary")

    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        st.subheader("Methodological Approach")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params["method"] = st.radio("Analysis Method", 
                                      options=["Simulation", "Hussey & Hughes Analytical"],
                                      help="Simulation uses Monte Carlo; Analytical uses Hussey & Hughes (2007) closed-form formulas",
                                      key="sw_method_binary")
        
        with col2:
            if params["method"] == "Hussey & Hughes Analytical":
                st.info("Using analytical formulas from Hussey & Hughes (2007)")
            else:
                st.info(f"Using {params['nsim']} Monte Carlo simulations")
        
        st.subheader("Correlation Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Within-cluster, within-period correlation (ICC):**")
            st.write(f"ICC = {params['icc']:.3f}")
            st.caption("Correlation between individuals in same cluster at same time")
        
        with col2:
            params["cluster_autocorr"] = st.slider("Cluster Autocorrelation (CAC)", 
                                                  min_value=0.0, max_value=0.9, value=0.0, step=0.01,
                                                  help="Within-cluster, between-period correlation - correlation within clusters over time",
                                                  key="sw_cac_binary")
            st.caption("Correlation within clusters across time periods")
            
        if params["cluster_autocorr"] > 0:
            st.info(f"""
            **Advanced correlation structure enabled:**
            - ICC (within-period): {params['icc']:.3f}
            - CAC (between-period): {params['cluster_autocorr']:.3f}
            """)
        else:
            st.info("Using standard correlation structure (ICC only)")

    # Display effect size information
    with st.expander("Effect Size Information", expanded=False):
        risk_diff = params["p_intervention"] - params["p_control"]
        if params["p_control"] > 0:
            risk_ratio = params["p_intervention"] / params["p_control"]
            odds_ratio = (params["p_intervention"] / (1 - params["p_intervention"])) / (params["p_control"] / (1 - params["p_control"]))
        else:
            risk_ratio = float('inf')
            odds_ratio = float('inf')
        
        st.write(f"**Risk Difference:** {risk_diff:.3f}")
        if risk_ratio != float('inf'):
            st.write(f"**Risk Ratio:** {risk_ratio:.3f}")
            st.write(f"**Odds Ratio:** {odds_ratio:.3f}")

    # Display study design summary
    with st.expander("Study Design Summary", expanded=False):
        total_n = params["clusters"] * params["steps"] * params["individuals_per_cluster"]
        control_periods = params["clusters"] * 1  # Baseline only
        intervention_periods = params["clusters"] * (params["steps"] - 1)
        
        st.write(f"**Total Sample Size:** {total_n:,} individuals")
        st.write(f"**Total Cluster-Periods:** {params['clusters'] * params['steps']}")
        st.write(f"**Control Periods:** {control_periods}")
        st.write(f"**Intervention Periods:** {intervention_periods}")
        st.write(f"**Design Effect (approx):** {1 + (params['individuals_per_cluster'] - 1) * params['icc']:.2f}")

    # Add button to preview design layout (safe visualization)
    if st.button("🔍 Preview Design Layout", help="Generate a visual preview of the stepped wedge design"):
        try:
            display_stepped_wedge_design(params["clusters"], params["steps"])
        except Exception as e:
            st.error(f"Could not generate design preview: {str(e)}")

    return params


def calculate_stepped_wedge_continuous(params):
    """
    Calculate power for stepped wedge design with continuous outcome.
    
    Args:
        params: Dictionary of parameters from the UI
        
    Returns:
        dict: Results from the power calculation
    """
    try:
        # Extract parameters
        clusters = int(params["clusters"])
        steps = int(params["steps"])
        individuals_per_cluster = int(params["individuals_per_cluster"])
        icc = float(params["icc"])
        cluster_autocorr = float(params.get("cluster_autocorr", 0.0))
        treatment_effect = float(params["treatment_effect"])
        std_dev = float(params["std_dev"])
        alpha = float(params["alpha"])
        method = params.get("method", "Simulation")
        
        if method == "Hussey & Hughes Analytical":
            # Use analytical method
            result = hussey_hughes_power_continuous(
                clusters=clusters,
                steps=steps,
                individuals_per_cluster=individuals_per_cluster,
                icc=icc,
                cluster_autocorr=cluster_autocorr,
                treatment_effect=treatment_effect,
                std_dev=std_dev,
                alpha=alpha
            )
            result["method"] = "Hussey & Hughes Analytical"
        else:
            # Use simulation method (extended to include cluster_autocorr if > 0)
            nsim = int(params["nsim"])
            
            if cluster_autocorr > 0:
                st.warning("⚠️ Cluster autocorrelation > 0 detected. Current simulation method uses simplified correlation structure. Consider using Hussey & Hughes Analytical method for more accurate results with complex correlation structures.")
            
            result = simulate_continuous(
                clusters=clusters,
                steps=steps,
                individuals_per_cluster=individuals_per_cluster,
                icc=icc,
                treatment_effect=treatment_effect,
                std_dev=std_dev,
                nsim=nsim,
                alpha=alpha
            )
            result["method"] = "Simulation"
            
            # Add cluster autocorr to parameters for display
            result["parameters"]["cluster_autocorr"] = cluster_autocorr
        
        # Add common metadata
        result["design_type"] = "Stepped Wedge"
        result["outcome_type"] = "Continuous"
        
        return result
        
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")
        return None


def calculate_stepped_wedge_binary(params):
    """
    Calculate power for stepped wedge design with binary outcome.
    
    Args:
        params: Dictionary of parameters from the UI
        
    Returns:
        dict: Results from the power calculation
    """
    try:
        # Extract parameters
        clusters = int(params["clusters"])
        steps = int(params["steps"])
        individuals_per_cluster = int(params["individuals_per_cluster"])
        icc = float(params["icc"])
        cluster_autocorr = float(params.get("cluster_autocorr", 0.0))
        p_control = float(params["p_control"])
        p_intervention = float(params["p_intervention"])
        alpha = float(params["alpha"])
        method = params.get("method", "Simulation")
        
        if method == "Hussey & Hughes Analytical":
            # Use analytical method
            result = hussey_hughes_power_binary(
                clusters=clusters,
                steps=steps,
                individuals_per_cluster=individuals_per_cluster,
                icc=icc,
                cluster_autocorr=cluster_autocorr,
                p_control=p_control,
                p_intervention=p_intervention,
                alpha=alpha
            )
            result["method"] = "Hussey & Hughes Analytical"
        else:
            # Use simulation method
            nsim = int(params["nsim"])
            
            if cluster_autocorr > 0:
                st.warning("⚠️ Cluster autocorrelation > 0 detected. Current simulation method uses simplified correlation structure. Consider using Hussey & Hughes Analytical method for more accurate results with complex correlation structures.")
            
            result = simulate_binary(
                clusters=clusters,
                steps=steps,
                individuals_per_cluster=individuals_per_cluster,
                icc=icc,
                p_control=p_control,
                p_intervention=p_intervention,
                nsim=nsim,
                alpha=alpha
            )
            result["method"] = "Simulation"
            
            # Add cluster autocorr to parameters for display
            result["parameters"]["cluster_autocorr"] = cluster_autocorr
        
        # Add common metadata
        result["design_type"] = "Stepped Wedge"
        result["outcome_type"] = "Binary"
        
        return result
        
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")
        return None


def generate_cli_code_stepped_wedge_continuous(params):
    """
    Generate CLI code for stepped wedge continuous outcome calculation.
    
    Args:
        params: Dictionary of parameters from the UI
        
    Returns:
        str: CLI command string
    """
    clusters = params["clusters"]
    steps = params["steps"]
    individuals_per_cluster = params["individuals_per_cluster"]
    icc = params["icc"]
    cluster_autocorr = params.get("cluster_autocorr", 0.0)
    treatment_effect = params["treatment_effect"]
    std_dev = params["std_dev"]
    alpha = params["alpha"]
    method = params.get("method", "Simulation")
    nsim = params.get("nsim", 1000)
    
    if method == "Hussey & Hughes Analytical":
        cli_code = f"""# Stepped Wedge Cluster RCT - Continuous Outcome (Analytical Method)
from core.designs.stepped_wedge.analytical import hussey_hughes_power_continuous

# Calculate power using Hussey & Hughes (2007) analytical method
result = hussey_hughes_power_continuous(
    clusters={clusters},
    steps={steps},
    individuals_per_cluster={individuals_per_cluster},
    icc={icc},
    cluster_autocorr={cluster_autocorr},
    treatment_effect={treatment_effect},
    std_dev={std_dev},
    alpha={alpha}
)

# Display results
print("="*60)
print("STEPPED WEDGE POWER ANALYSIS - CONTINUOUS OUTCOME")
print("="*60)
print(f"Method: Hussey & Hughes (2007) Analytical")
print(f"Clusters: {clusters}")
print(f"Time steps: {steps}")
print(f"Individuals per cluster per period: {individuals_per_cluster}")
print(f"Total sample size: {{result['parameters']['total_n']:,}}")
print(f"ICC (within-cluster, within-period): {icc:.3f}")"""
        
        if cluster_autocorr > 0:
            cli_code += f'\nprint(f"CAC (within-cluster, between-period): {cluster_autocorr:.3f}")'
            
        cli_code += f"""
print(f"Treatment effect: {treatment_effect}")
print(f"Standard deviation: {std_dev}")
print(f"Alpha: {alpha}")
print("-"*60)
print(f"POWER: {{result['power']:.3f}} ({{result['power']*100:.1f}}%)")
print(f"Standard error of treatment effect: {{result['se_treatment_effect']:.4f}}")"""
        
        if cluster_autocorr > 0:
            cli_code += '\nprint(f"Correlation adjustment factor: {result[\'correlation_adjustment\']:.3f}")'
            
        cli_code += '\nprint("="*60)'
        
    else:
        cli_code = f"""# Stepped Wedge Cluster RCT - Continuous Outcome (Simulation Method)
from core.designs.stepped_wedge.simulation import simulate_continuous
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Calculate power using Monte Carlo simulation
result = simulate_continuous(
    clusters={clusters},
    steps={steps},
    individuals_per_cluster={individuals_per_cluster},
    icc={icc},
    treatment_effect={treatment_effect},
    std_dev={std_dev},
    nsim={nsim},
    alpha={alpha}
)

# Display results
print("="*60)
print("STEPPED WEDGE POWER ANALYSIS - CONTINUOUS OUTCOME")
print("="*60)
print(f"Method: Monte Carlo Simulation")
print(f"Clusters: {clusters}")
print(f"Time steps: {steps}")
print(f"Individuals per cluster per period: {individuals_per_cluster}")
print(f"Total sample size: {{result['parameters']['total_n']:,}}")
print(f"ICC: {icc:.3f}")
print(f"Treatment effect: {treatment_effect}")
print(f"Standard deviation: {std_dev}")
print(f"Alpha: {alpha}")
print(f"Number of simulations: {nsim:,}")
print("-"*60)
print(f"POWER: {{result['power']:.3f}} ({{result['power']*100:.1f}}%)")
print(f"Mean p-value: {{result['mean_p_value']:.4f}}")
print(f"Median p-value: {{result['median_p_value']:.4f}}")
print("="*60)"""
    
    return cli_code


def generate_cli_code_stepped_wedge_binary(params):
    """
    Generate CLI code for stepped wedge binary outcome calculation.
    
    Args:
        params: Dictionary of parameters from the UI
        
    Returns:
        str: CLI command string
    """
    clusters = params["clusters"]
    steps = params["steps"]
    individuals_per_cluster = params["individuals_per_cluster"]
    icc = params["icc"]
    cluster_autocorr = params.get("cluster_autocorr", 0.0)
    p_control = params["p_control"]
    p_intervention = params["p_intervention"]
    alpha = params["alpha"]
    method = params.get("method", "Simulation")
    nsim = params.get("nsim", 1000)
    
    # Calculate effect measures for display
    risk_diff = p_intervention - p_control
    if p_control > 0:
        risk_ratio = p_intervention / p_control
        odds_ratio = (p_intervention / (1 - p_intervention)) / (p_control / (1 - p_control))
    else:
        risk_ratio = float('inf')
        odds_ratio = float('inf')
    
    if method == "Hussey & Hughes Analytical":
        cli_code = f"""# Stepped Wedge Cluster RCT - Binary Outcome (Analytical Method)
from core.designs.stepped_wedge.analytical import hussey_hughes_power_binary

# Calculate power using Hussey & Hughes (2007) analytical method with arcsine transformation
result = hussey_hughes_power_binary(
    clusters={clusters},
    steps={steps},
    individuals_per_cluster={individuals_per_cluster},
    icc={icc},
    cluster_autocorr={cluster_autocorr},
    p_control={p_control},
    p_intervention={p_intervention},
    alpha={alpha}
)

# Display results
print("="*60)
print("STEPPED WEDGE POWER ANALYSIS - BINARY OUTCOME")
print("="*60)
print(f"Method: Hussey & Hughes (2007) Analytical (Arcsine Transformation)")
print(f"Clusters: {clusters}")
print(f"Time steps: {steps}")
print(f"Individuals per cluster per period: {individuals_per_cluster}")
print(f"Total sample size: {{result['parameters']['total_n']:,}}")
print(f"ICC (within-cluster, within-period): {icc:.3f}")"""
        
        if cluster_autocorr > 0:
            cli_code += f'\nprint(f"CAC (within-cluster, between-period): {cluster_autocorr:.3f}")'
            
        cli_code += f"""
print(f"Control proportion: {p_control:.3f}")
print(f"Intervention proportion: {p_intervention:.3f}")
print(f"Risk difference: {risk_diff:.3f}")"""
        
        if risk_ratio != float('inf'):
            cli_code += f'\nprint(f"Risk ratio: {risk_ratio:.3f}")'
        if odds_ratio != float('inf'):
            cli_code += f'\nprint(f"Odds ratio: {odds_ratio:.3f}")'
            
        cli_code += f"""
print(f"Alpha: {alpha}")
print("-"*60)
print(f"POWER: {{result['power']:.3f}} ({{result['power']*100:.1f}}%)")
print(f"Standard error of treatment effect: {{result['se_treatment_effect']:.4f}}")"""
        
        if cluster_autocorr > 0:
            cli_code += '\nprint(f"Correlation adjustment factor: {result[\'correlation_adjustment\']:.3f}")'
            
        cli_code += '\nprint("="*60)'
        
    else:
        cli_code = f"""# Stepped Wedge Cluster RCT - Binary Outcome (Simulation Method)
from core.designs.stepped_wedge.simulation import simulate_binary
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Calculate power using Monte Carlo simulation
result = simulate_binary(
    clusters={clusters},
    steps={steps},
    individuals_per_cluster={individuals_per_cluster},
    icc={icc},
    p_control={p_control},
    p_intervention={p_intervention},
    nsim={nsim},
    alpha={alpha}
)

# Display results
print("="*60)
print("STEPPED WEDGE POWER ANALYSIS - BINARY OUTCOME")
print("="*60)
print(f"Method: Monte Carlo Simulation")
print(f"Clusters: {clusters}")
print(f"Time steps: {steps}")
print(f"Individuals per cluster per period: {individuals_per_cluster}")
print(f"Total sample size: {{result['parameters']['total_n']:,}}")
print(f"ICC: {icc:.3f}")
print(f"Control proportion: {p_control:.3f}")
print(f"Intervention proportion: {p_intervention:.3f}")
print(f"Risk difference: {risk_diff:.3f}")"""
        
        if risk_ratio != float('inf'):
            cli_code += f'\nprint(f"Risk ratio: {risk_ratio:.3f}")'
        if odds_ratio != float('inf'):
            cli_code += f'\nprint(f"Odds ratio: {odds_ratio:.3f}")'
            
        cli_code += f"""
print(f"Alpha: {alpha}")
print(f"Number of simulations: {nsim:,}")
print("-"*60)
print(f"POWER: {{result['power']:.3f}} ({{result['power']*100:.1f}}%)")
print(f"Mean p-value: {{result['mean_p_value']:.4f}}")
print(f"Median p-value: {{result['median_p_value']:.4f}}")
print("="*60)"""
    
    return cli_code