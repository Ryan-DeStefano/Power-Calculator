import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t
from utils.two_sample_t import *
from utils.utils import *
from utils.one_sample_t import *


# --- Streamlit Interface ---
st.set_page_config(page_title="Power Calculator")
st.title("Power Calculator")

# Tabs
tab = st.tabs(["One-Sample T-Test", "Two-Sample T-Test"])

with tab[0]:

    with st.expander("What is Effect Size (Cohen's d)?"):
        st.markdown("""
            **Effect size**, measured by **Cohen’s d**, tells us how big the true difference is between a true population mean and a known or hypothesized population mean.
        """)

        st.markdown("The formula is:")
        st.latex(r"d = \frac{\mu - \mu_0}{\sigma}")

        st.markdown("""
            Where:
            - **μ** is the population mean
            - **μ₀** is the population mean you're testing against  
            - **σ** is the population standard deviation

            ---

            When you enter an effect size you're saying:

            > “This is the size of the difference I expect or want to be able to detect between my sample and the population.”

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


    col1, col2 = st.columns(2)

    with col1:
        alpha = round(st.number_input(
            label="Significance Level (alpha)", 
            min_value=0.001, 
            max_value=0.3, 
            value=0.05, 
            step=0.01, 
            format="%.2f", 
            key="alpha_one_sample_t",
            help="The probability of rejecting the null hypothesis when it is true (Type I error rate)."),
            4)

    with col2:
        alternative = st.selectbox(
            label="Test Type", 
            options=['two-sided', 'larger', 'smaller'], 
            help="Specifies the alternative hypothesis:\n"
             "- two-sided: tests if the sample mean is different from the test mean (μ₀)\n"
             "- larger: tests if the sample mean is greater than the test mean (μ₀)\n"
             "- smaller: tests if the sample mean is less than the test mean (μ₀)",
            key="alternative_one_sample_t")

    # Display null and alternative hypotheses
    st.markdown("### Hypotheses")

    null_hypothesis = "H₀: μ = μ₀"
    if alternative == "two-sided":
        alt_hypothesis = "H₁: μ ≠ μ₀"
    elif alternative == "larger":
        alt_hypothesis = "H₁: μ > μ₀"
    elif alternative == "smaller":
        alt_hypothesis = "H₁: μ < μ₀"

    st.latex(null_hypothesis)
    st.latex(alt_hypothesis)

    # Power or sample size mode
    mode = st.radio(
        label="What do you want to calculate?", 
        options=["Power", "Sample Size", "Effect Size"],
        key="mode_one_sample_t"
    )

    if mode == "Power":

        col1, col2 = st.columns(2)

        with col1:
            n = input_sample_size(group_label="Sample Size", key="n_power_one_sample", default=50)
        with col2:
            effect_size = input_effect_size(key="effect_power_one_sample", default=0.5)

        power, df = power_one_sample_ttest(
            effect_size=effect_size, 
            n=n, 
            alpha=alpha,
            alternative=alternative)
     
        warning_msg = validate_effect_size_direction_power(
            effect_size=effect_size, 
            alternative=alternative)

        if warning_msg:
            st.warning(warning_msg)

        if st.button("Calculate Power", key="calc_power_one_sample"):
            st.success(f"Estimated Power: **{power:.3f}**")

            plot_power_curve_with_distributions_one_sample_ttest(
                effect_size=effect_size, 
                n=n, 
                df=df, 
                alpha=alpha, 
                alternative=alternative)

    elif mode == "Sample Size":
        
        col1, col2 = st.columns(2)

        with col1:
            power_target = input_power(key="power_sample_one_sample", default=0.8)
        with col2:
            effect_size = input_effect_size(key="effect_sample_one_sample", default=0.5)

        warning_msg = validate_effect_size_direction(
            effect_size=effect_size,
            alternative=alternative)
        
        if warning_msg:
            st.warning(warning_msg)
        
        valid_result = False

        try:
            n_required = sample_size_one_sample_ttest(
                effect_size=effect_size, 
                power_target=power_target, 
                alpha=alpha, 
                alternative=alternative)
            valid_result = True
        except ValueError as e:
            st.warning(str(e))
            
        if st.button("Calculate Sample Size", key="calc_sample_one_sample") and valid_result:
            st.success(f"Required Sample Size: **{int(n_required)}**")
        
            if warning_msg:
                pass
            else:
                plot_power_curve_one_sample_ttest(
                    effect_size=effect_size, 
                    alpha=alpha,
                    power_target=power_target,
                    alternative=alternative, 
                    max_n=n_required*3)
                    
    elif mode == "Effect Size":

        col1, col2 = st.columns(2)

        with col1:
            n = input_sample_size(group_label="Sample Size", key="n_effect_one_sample", default=50)
        with col2:
            power_target = input_power(key="power_effect_one_sample", default=0.8)

        effect_size = effect_size_one_sample_ttest(
            n=n, 
            power_target=power_target, 
            alpha=alpha, 
            alternative=alternative)
                
        if st.button("Calculate Effect Size", key="calc_effect_one_sample"):
            st.success(f"Effect Size: **{effect_size:.3f}**")

with tab[1]:
    
    with st.expander("What is Effect Size (Cohen's d)?"):
        st.markdown("""
            **Effect size**, measured by **Cohen’s d**, tells us how big the true difference is between two groups, like a treatment group and a control group.""")
        
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

    col1, col2, col3 = st.columns(3)

    with col1:
        alpha = round(st.number_input(
            label="Significance Level (alpha)", 
            min_value=0.001, 
            max_value=0.3, 
            value=0.05, 
            step=0.01, 
            format="%.2f", 
            key="alpha_two_sample_t",
            help="The probability of rejecting the null hypothesis when it is true (Type I error rate)."),
            4)

    with col2:
        alternative = st.selectbox(
            label="Test Type", 
            options=['two-sided', 'larger', 'smaller'], 
            help="Specifies the alternative hypothesis type:\n"
            "- two-sided: tests for difference in either direction\n"
            "- larger: tests if group 1 mean is greater than group 2\n"
            "- smaller: tests if group 1 mean is less than group 2",
            key="alternative_two_sample_t")

    with col3:
        variance_type = st.radio(
            label="Assume equal variances?",
            options=["Equal Variances", "Unequal Variances"],
            help="Select whether to assume equal population variances. This affects how effect size is specified and power/sample size is calculated. " \
                "Only assume unequal variances if you have prior knowledge on what the population mean and standard deviations are.",
            key="variance_two_sample_t")

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
    mode = st.radio(
        label="What do you want to calculate?", 
        options=["Power", "Sample Size", "Effect Size"],
        key="mode_two_sample_t"
    )

    if mode == "Power":

        col1, col2 = st.columns(2)

        if variance_type == "Equal Variances":
            with col1:
                n1 = input_sample_size(group_label="Group 1 Sample Size", key="n1_power_two_sample",default=50)
                effect_size = input_effect_size(key="effect_power_two_sample", default=0.5)
            with col2:
                n2 = input_sample_size(group_label="Group 2 Sample Size", key="n2_power_two_sample", default=50)

            power, df = power_two_sample_ttest_equal_variance(
                effect_size=effect_size, 
                n1=n1, 
                n2=n2, 
                alpha=alpha,
                alternative=alternative)

        else:
            with col1:
                n1 = input_sample_size(group_label="Group 1 Sample Size", key="n1_power_two_sample2", default=50)
                mu1 = input_mean(group_label="₁", key="mu1_power_two_sample", default=0.0)
                s1 = input_std(group_label="₁", key="s1_power_two_sample", default=1.0)
            with col2:
                n2 = input_sample_size(group_label="Group 2 Sample Size", key="n2_power_two_sample2", default=50)            
                mu2 = input_mean(group_label="₂", key="mu2_power_two_sample", default=0.0)
                s2 = input_std(group_label="₂", key="s2_power_two_sample", default=1.0)

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

        if st.button("Calculate Power", key="calc_power_two_sample"):
            st.success(f"Estimated Power: **{power:.3f}**")

            plot_power_curve_with_distributions_two_sample_ttest(
                effect_size=effect_size, 
                n1=n1, 
                n2=n2, 
                df=df, 
                alpha=alpha, 
                alternative=alternative)

    elif mode == "Sample Size":

        col1, col2 = st.columns(2)

        with col1:
            power_target = input_power(key="power_sample_two_sample", default=0.8)
        with col2:
            sample_size_ratio = input_sample_ratio(key="ratio_sample_two_sample", default=1.0)

        if variance_type == "Equal Variances":

            with col1:
                effect_size = input_effect_size(key="effect_sample_two_sample", default=0.5)

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

            with col1:
                mu1 = input_mean(group_label="₁", key="mu1_sample_two_sample", default=0.0)
                s1 = input_std(group_label="₁", key="s1_sample_two_sample", default=1.0)
            with col2:
                mu2 = input_mean(group_label="₂", key="mu2_sample_two_sample", default=0.0)
                s2 = input_std(group_label="₂", key="s2_sample_two_sample", default=1.0)
            
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
            
        if st.button("Calculate Sample Size", key="calc_sample_two_sample") and valid_result:
            st.success(f"Required Sample Sizes:\n- Group 1: **{int(n1_required)}**\n- Group 2: **{int(n2_required)}**")
        
            if warning_msg:
                pass
            else:

                if variance_type == "Equal Variances":
                    plot_power_curve_two_sample_ttest_equal_variance(
                        effect_size=effect_size, 
                        alpha=alpha,
                        power_target=power_target,
                        alternative=alternative, 
                        allocation_ratio=sample_size_ratio, 
                        max_n1=n1_required*3)
                    
                else:
                    plot_power_curve_two_sample_ttest_unequal_variance(
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

        col1, col2 = st.columns(2)

        with col1:
            n1 = input_sample_size(group_label="Group 1 Sample Size", key="n1_effect_two_sample", default=50)
            power_target = input_power(key="power_effect_two_sample", default=0.8)
        with col2:
            n2 = input_sample_size(group_label="Group 2 Sample Size", key="n2_effect_two_sample", default=50)
        
        effect_size = effect_size_two_sample_ttest(n1, n2, power_target, alpha, alternative)

        if st.button("Calculate Effect Size", key="calc_effect_two_sample"):
            st.success(f"Effect Size: **{effect_size:.3f}**")