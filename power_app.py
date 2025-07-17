import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t
from utils.two_sample_t import *
from utils.utils import *


# --- Streamlit Interface ---
st.set_page_config(page_title="Power Calculator")
st.title("Two Sample t-Test Power Calculator")

# Sidebar input options
st.sidebar.header("Input Parameters")

alpha = round(st.sidebar.number_input(
    label="Significance Level (alpha)", 
    min_value=0.001, 
    max_value=0.3, 
    value=0.05, 
    step=0.01, 
    format="%.2f", 
    help="The probability of rejecting the null hypothesis when it is true (Type I error rate)."),
    4)

alternative = st.sidebar.selectbox(
    label="Test Type", 
    options=['two-sided', 'larger', 'smaller'], 
    help="Specifies the alternative hypothesis type:\n"
    "- two-sided: tests for difference in either direction\n"
    "- larger: tests if group 1 mean is greater than group 2\n"
    "- smaller: tests if group 1 mean is less than group 2")

variance_type = st.sidebar.radio(
    label="Assume equal variances?",
    options=["Equal Variances", "Unequal Variances"],
    help="Select whether to assume equal population variances. This affects how effect size is specified and power/sample size is calculated. " \
         "Only assume unequal variances if you have prior knowledge on what the population mean and standard deviations are.")

with st.expander("What is Effect Size (Cohen's d)?"):
    st.markdown("""
        **Effect size**, measured by **Cohen’s d**, tells us **how big the true difference is between two groups**, like a treatment group and a control group.""")
    
    st.markdown("The formula is:")
    st.latex(r"d = \frac{\mu_1 - \mu_2}{\sigma}")

    st.markdown("""
        Where:
        - **μ₁** and **μ₂** are the means of the two groups  
        - **σ** is the pooled standard deviation 

        ---

        When you enter an effect size you're saying:

        > “This is the size of the difference I expect or want to be able to detect between the two groups.”

        Effect size is measured in **standard deviation units**, so:
        - A **small effect** is a small difference (e.g., d = 0.2)
        - A **medium effect** is a moderate difference (e.g., d = 0.5)
        - A **large effect** is a large difference (e.g., d = 0.8 or more)

        ---

        **In a power calculation**, the effect size helps answer:
        > _How many samples do I need to detect this difference with high confidence?_

        Or, for calculating power:
        > _How likely am I to detect a difference of this size with my current sample size?_

        So the **effect size you input** defines the size of difference you're trying to detect **with certainty (power)**.
        """)


# Display null and alternative hypotheses
st.markdown("### Hypotheses")

null_hypothesis = "H₀: μ₁ = μ₂"
if alternative == "two-sided":
    alt_hypothesis = "H₁: μ₁ ≠ μ₂"
elif alternative == "larger":
    alt_hypothesis = "H₁: μ₁ > μ₂"
elif alternative == "smaller":
    alt_hypothesis = "H₁: μ₁ < μ₂"

st.latex(null_hypothesis)
st.latex(alt_hypothesis)

# Power or sample size mode
mode = st.radio("What do you want to calculate?", options=["Power", "Sample Size", "Effect Size"])

if mode == "Power":

    n1 = input_sample_size(group_label="1", default=50)
    n2 = input_sample_size(group_label="2", default=50)

    if variance_type == "Equal Variances":

        effect_size = input_effect_size(default=0.5)

        power, df = power_two_sample_ttest_equal_variance(
            effect_size=effect_size, 
            n1=n1, 
            n2=n2, 
            alpha=alpha,
            alternative=alternative)

    else:

        mu1 = input_mean(group_label="₁", default=0.0)
        mu2 = input_mean(group_label="₂", default=0.0)
        s1 = input_std(group_label="₁", default=1.0)
        s2 = input_std(group_label="₂", default=1.0)

        power, effect_size, df = power_two_sample_ttest_unequal_variance(
            n1=n1, 
            n2=n2, 
            mu1=mu1, 
            mu2=mu2, 
            s1=s1, 
            s2=s2, 
            alpha=alpha, 
            alternative=alternative)
        
        effect_size = round(effect_size, 4)

    warning_msg = validate_effect_size_direction_power(
        effect_size=effect_size, 
        alternative=alternative)

    if warning_msg:
        st.warning(warning_msg)

    if st.button("Calculate Power"):
        st.success(f"Estimated Power: **{power:.3f}**")

        plot_power_curve_with_distributions(
            effect_size=effect_size, 
            n1=n1, 
            n2=n2, 
            df=df, 
            alpha=alpha, 
            alternative=alternative)

elif mode == "Sample Size":

    power_target = input_power(default=0.8)
    sample_size_ratio = input_sample_ratio(default=1.0)

    if variance_type == "Equal Variances":

        effect_size = input_effect_size(default=0.5)

        warning_msg = validate_effect_size_direction_sample_equal_variance(
            effect_size=effect_size,
            alternative=alternative)
        
        if warning_msg:
            st.warning(warning_msg)
        
        valid_result = False

        try:
            n1_required, n2_required = sample_size_two_sample_ttest_equal_variance(
                effect_size=effect_size, 
                power_target=power_target, 
                alpha=alpha, 
                alternative=alternative, 
                allocation_ratio=sample_size_ratio)
            valid_result = True
        except ValueError as e:
            st.warning(str(e))
            
    else:

        mu1 = input_mean(group_label="₁", default=0.0)
        mu2 = input_mean(group_label="₂", default=0.0)
        s1 = input_std(group_label="₁", default=1.0)
        s2 = input_std(group_label="₂", default=1.0)
        
        warning_msg = validate_effect_size_direction_sample_unequal_variance(
            mu1=mu1,
            mu2=mu2,
            s1=s1,
            s2=s2,
            alternative=alternative)
        
        if warning_msg:
            st.warning(warning_msg)

        valid_result = False

        try:
            n1_required, n2_required = sample_size_two_sample_ttest_unequal_variance(
                mu1=mu1,
                mu2=mu2,
                s1=s1,
                s2=s2, 
                power_target=power_target, 
                alpha=alpha, 
                alternative=alternative, 
                allocation_ratio=sample_size_ratio)
            valid_result=True
        except ValueError as e:
            st.warning(str(e))
        
    if st.button("Calculate Sample Size") and valid_result:
        st.success(f"Required Sample Sizes:\n- Group 1: **{int(n1_required)}**\n- Group 2: **{int(n2_required)}**")
    
        if warning_msg:
            pass
        else:

            if variance_type == "Equal Variances":
                plot_power_curve_equal_variance(
                    effect_size=effect_size, 
                    alpha=alpha,
                    power_target=power_target,
                    alternative=alternative, 
                    allocation_ratio=sample_size_ratio, 
                    max_n1=n1_required*3)
                
            else:
                plot_power_curve_unequal_variance(
                    mu1=mu1,
                    mu2=mu2,
                    s1=s1,
                    s2=s2, 
                    alpha=alpha,
                    power_target=power_target,
                    alternative=alternative, 
                    allocation_ratio=sample_size_ratio, 
                    max_n1=n1_required*3)
                
elif mode == "Effect Size":

    n1 = input_sample_size(group_label="1", default=50)
    n2 = input_sample_size(group_label="2", default=50)
    
    power_target = input_power(default=0.8)

    effect_size = effect_size_two_sample_ttest(n1, n2, power_target, alpha, alternative)
    st.success(f"Effect Size: **{effect_size:.3f}**")