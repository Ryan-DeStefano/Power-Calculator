# streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t



def validate_effect_size_direction_power(effect_size, alternative):
    if alternative == 'larger' and effect_size < 0:
        return "**WARNING:** Effect size is negative but test type is 'larger'. Consider flipping the sign or changing the test direction."
    elif alternative == 'smaller' and effect_size > 0:
        return "**WARNING:** Effect size is positive but test type is 'smaller'. Consider flipping the sign or changing the test direction."
    return None

def validate_effect_size_direction_sample_equal_variance(effect_size, alternative):
    if alternative == 'larger' and effect_size < 0:
        return "**WARNING:** Effect size is negative but test type is 'larger'. Sample sizes are not accurate, flip the direction of one of effect size or test type."
    elif alternative == 'smaller' and effect_size > 0:
        return "**WARNING:** Effect size is positive but test type is 'smaller'. Sample sizes are not accurate, flip the direction of one of effect size or test type."
    return None

def validate_effect_size_direction_sample_unequal_variance(mu1, mu2, s1, s2, alternative):

    pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
    effect_size = (mu1 - mu2) / pooled_sd 

    if alternative == 'larger' and effect_size < 0:
        return "**WARNING:** Effect size is negative but test type is 'larger'. Sample sizes are not accurate, flip the direction of one of effect size or test type."
    elif alternative == 'smaller' and effect_size > 0:
        return "**WARNING:** Effect size is positive but test type is 'smaller'. Sample sizes are not accurate, flip the direction of one of effect size or test type."
    return None

# --- Power Calculation Functions ---
def power_two_sample_ttest_equal_variance(effect_size, n1, n2, alpha=0.05, alternative='two-sided'):

    df = n1 + n2 - 2
    ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))

    t_crit = stats.t.ppf(1 - alpha / 2, df) if alternative == 'two-sided' else stats.t.ppf(1 - alpha, df)

    if alternative == 'two-sided':
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    elif alternative == 'larger':
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    elif alternative == 'smaller':
        power = stats.nct.cdf(-t_crit, df, ncp)
    else:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
    
    return power, df

def power_two_sample_ttest_unequal_variance(n1, n2, mu1, mu2, s1, s2, alpha=0.05, alternative='two-sided'):

    numerator = (s1**2 / n1 + s2**2 / n2)**2
    denominator = ((s1**2 / n1)**2) / (n1 - 1) + ((s2**2 / n2)**2) / (n2 - 1)
    df = np.floor(numerator / denominator)

    pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
    effect_size = (mu1 - mu2) / pooled_sd
    
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    ncp = (mu1 - mu2) / se

    t_crit = stats.t.ppf(1 - alpha / 2, df) if alternative == 'two-sided' else stats.t.ppf(1 - alpha, df)

    if alternative == 'two-sided':
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    elif alternative == 'larger':
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    elif alternative == 'smaller':
        power = stats.nct.cdf(-t_crit, df, ncp)
    else:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
    
    if np.isnan(power):
        power = 0.0
    
    return power, effect_size, df

def plot_power_curve_with_distributions(effect_size, n1, n2, df, alpha=0.05, alternative='two-sided', normal_approx_threshold=350):
    ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))

    use_normal_approx = df > normal_approx_threshold

    # x-axis range
    x = np.linspace(-6, 6 + ncp, 1000)

    # Null distribution (central t)
    null_pdf = stats.t.pdf(x, df) if not use_normal_approx else stats.norm.pdf(x, 0, 1)

    # Alternative distribution
    alt_pdf = stats.nct.pdf(x, df, ncp) if not use_normal_approx else stats.norm.pdf(x, ncp, 1)

    # Critical values
    if alternative == 'two-sided':
        t_crit_low = stats.t.ppf(alpha / 2, df) if not use_normal_approx else stats.norm.ppf(alpha / 2)
        t_crit_high = stats.t.ppf(1 - alpha / 2, df) if not use_normal_approx else stats.norm.ppf(1 - alpha / 2)
    else:
        t_crit = stats.t.ppf(1 - alpha, df) if not use_normal_approx else stats.norm.ppf(1 - alpha)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, null_pdf, label='Null Distribution', color='blue')
    plt.plot(x, alt_pdf, label='Alternative Distribution', color='orange')

    # Shading rejection regions and power regions
    if alternative == 'two-sided':
        plt.fill_between(x, 0, null_pdf, where=(x <= t_crit_low) | (x >= t_crit_high), color='blue', alpha=0.3, label='Rejection Region (α/2)')
        plt.fill_between(x, 0, alt_pdf, where=(x <= t_crit_low) | (x >= t_crit_high), color='red', alpha=0.3, label='Power')
        plt.axvline(t_crit_low, color='black', linestyle='--', label='-t critical')
        plt.axvline(t_crit_high, color='black', linestyle='--', label='t critical')
    elif alternative == 'larger':
        plt.fill_between(x, 0, null_pdf, where=(x >= t_crit), color='blue', alpha=0.3, label='Rejection Region (α)')
        plt.fill_between(x, 0, alt_pdf, where=(x >= t_crit), color='red', alpha=0.3, label='Power')
        plt.axvline(t_crit, color='black', linestyle='--', label='t critical')
    elif alternative == 'smaller':
        plt.fill_between(x, 0, null_pdf, where=(x <= -t_crit), color='blue', alpha=0.3, label='Rejection Region (α)')
        plt.fill_between(x, 0, alt_pdf, where=(x <= -t_crit), color='red', alpha=0.3, label='Power')
        plt.axvline(-t_crit, color='black', linestyle='--', label='t critical')

    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Two-Sample t-Test Power Curve\nEffect Size={effect_size}, α={alpha}, n1={n1}, n2={n2}')
    plt.xlabel('t-value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())


# --- Sample Size Solver ---
def sample_size_two_sample_ttest_equal_variance(effect_size, power_target=0.8, alpha=0.05,
                                                alternative='two-sided', allocation_ratio=2.0):
    
    if effect_size == 0:
        raise ValueError("**WARNING:** Effect size is zero, so sample size is infinite. Enter an effect size that is not 0.")
    
    z_alpha = norm.ppf(1 - alpha / 2) if alternative == 'two-sided' else norm.ppf(1 - alpha)
    z_beta = norm.ppf(power_target)

    # n1 is the size of the smaller (or reference) group
    n1 = ((z_alpha + z_beta) ** 2 * (1 + allocation_ratio)) / (effect_size ** 2)
    n2 = n1 / allocation_ratio

    return np.ceil(n1), np.ceil(n2)

def sample_size_two_sample_ttest_unequal_variance(mu1, mu2, s1, s2, power_target=0.8, alpha=0.05,
                                                  alternative='two-sided', allocation_ratio=2.0):
    
    if mu1 == mu2:
        raise ValueError("**WARNING:** Means are equal so effect size is zero. So, sample size is infinite. Enter means that are not equal.")
    
    z_alpha = norm.ppf(1 - alpha / 2) if alternative == 'two-sided' else norm.ppf(1 - alpha)
    z_beta = norm.ppf(power_target)

    # n1 is the size of the smaller (or reference) group
    n1 = ((z_alpha + z_beta) ** 2 * (s1 ** 2 + s2 ** 2 * allocation_ratio)) / ((mu1 - mu2) ** 2)
    n2 = n1 / allocation_ratio

    return np.ceil(n1), np.ceil(n2)

def plot_power_curve_equal_variance(effect_size, alpha=0.05, power_target=0.8, alternative='two-sided', allocation_ratio=1.0, max_n1=200):
    n1_values = np.arange(2, max_n1 + 1)
    power_values = []

    for n1 in n1_values:
        n2 = n1 / allocation_ratio
        power, df = power_two_sample_ttest_equal_variance(effect_size=effect_size, n1=n1, n2=n2, alpha=alpha, alternative=alternative)
        power_values.append(power)

    plt.figure(figsize=(10, 6))
    plt.plot(n1_values, power_values, label=f'Power', )
    plt.axhline(y=power_target, color='red', linestyle='--', label=f'Target Power = {power_target}')
    plt.xlabel('Sample Size of Group 1 (n1)')
    plt.ylabel('Power')
    plt.title('Power vs. Sample Size (Group 1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())

def plot_power_curve_unequal_variance(mu1, mu2, s1, s2, alpha=0.05, power_target=0.8, alternative='two-sided', allocation_ratio=1.0, max_n1=200):
    n1_values = np.arange(2, max_n1 + 1)
    power_values = []

    for n1 in n1_values:
        n2 = n1 / allocation_ratio
        power, effect_size, df = power_two_sample_ttest_unequal_variance(mu1=mu1, mu2=mu2, s1=s1, s2=s2, n1=n1, n2=n2, alpha=alpha, alternative=alternative)
        power_values.append(power)

    plt.figure(figsize=(10, 6))
    plt.plot(n1_values, power_values, label=f'Power', )
    plt.axhline(y=power_target, color='red', linestyle='--', label=f'Target Power = {power_target}')
    plt.xlabel('Sample Size of Group 1 (n1)')
    plt.ylabel('Power')
    plt.title('Power vs. Sample Size (Group 1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())


# --- Effect Size Solver ---
def effect_size_two_sample_ttest(n1, n2, power_target=0.8, alpha=0.05, alternative='two-sided'):
    df = n1 + n2 - 2

    if alternative == 'two-sided':
        t_alpha = t.ppf(1 - alpha / 2, df)
        t_beta = t.ppf(power_target, df)
    elif alternative == 'larger':
        t_alpha = t.ppf(1 - alpha, df)
        t_beta = t.ppf(power_target, df)
    else:
        t_alpha = -1 * t.ppf(1 - alpha, df)
        t_beta = -1 * t.ppf(power_target, df)

    se_term = np.sqrt(1/n1 + 1/n2)
    d_min = (t_alpha + t_beta) * se_term
    
    return d_min


# --- Streamlit Interface ---
# Set initial page config (optional, still good for favicon)
st.set_page_config(page_title="Power Calculator", page_icon="📊")

# Override the tab title with JS to remove the appended '· Streamlit'
st.markdown(
    """
    <script>
    document.title = "Power Calculator";
    </script>
    """,
    unsafe_allow_html=True,
)

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

    n1 = round(st.number_input(
        label="Group 1 Sample Size", 
        min_value=2, value=50), 
        4)
    n2 = round(st.number_input(
        label="Group 2 Sample Size", 
        min_value=2, 
        value=50), 
        4)

    if variance_type == "Equal Variances":

        effect_size = round(st.number_input(
            label="Effect Size (Cohen's d)", 
            min_value=-2.0, 
            max_value=2.0, 
            value=0.5, 
            step=0.01, 
            format="%.2f",
            help="See top of page for explanation of Cohen's d"), 
            4)

        power, df = power_two_sample_ttest_equal_variance(
            effect_size=effect_size, 
            n1=n1, 
            n2=n2, 
            alpha=alpha,
            alternative=alternative)

    else:

        mu1 = round(st.number_input(
            label="μ₁", 
            min_value=-1e10, 
            max_value=1e10, 
            value=0.0, 
            step=1.0, 
            format="%.2f", 
            help="Estimated mean of population 1"), 
            4)
        mu2 = round(st.number_input(
            label="μ₂", 
            min_value=-1e10, 
            max_value=1e10, 
            value=0.0, 
            step=1.0, 
            format="%.2f", 
            help="Estimated mean of population 2"), 
            4)
        s1 = round(st.number_input(
            label="s₁", 
            min_value=0.01, 
            max_value=1e10, 
            value=1.0, 
            step=1.0, 
            format="%.2f", 
            help="Estimated standard deviation of population 1"),
            4)
        s2 = round(st.number_input(
            label="s₂", 
            min_value=0.01, 
            max_value=1e10, 
            value=1.0, 
            step=1.0, 
            format="%.2f", 
            help="Estimated standard deviation of population 2"), 
            4)

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

    power_target = round(st.number_input(
        label="Desired Power", 
        min_value=0.01, 
        max_value=0.99, 
        value=0.8, 
        step=0.01,
        help="Statistical power. This is the probability of correctly rejecting the null hypothesis when it is false."), 
        4)
    
    sample_size_ratio = round(st.number_input(
        label="Ratio of Group 1 Size to Group 2 Size",
        min_value=0.01, 
        max_value=1000.0, 
        value=1.0, 
        step=0.1,
        help="The ratio n₁/n₂ of the two sample sizes. Use 1.0 for equal group sizes. For example, 2.0 means Group 1 has twice as many samples as Group 2."), 
        4)

    if variance_type == "Equal Variances":

        effect_size = round(st.number_input(
            label="Effect Size (Cohen's d)", 
            min_value=-2.0, 
            max_value=2.0, 
            value=0.5, 
            step=0.01, 
            format="%.2f",
            help="See top of page for explanation of Cohen's d"), 
            4)

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

        mu1 = round(st.number_input(
            label="μ₁", 
            min_value=-1e10, 
            max_value=1e10, 
            value=0.0, 
            step=1.0, 
            format="%.2f", 
            help="Estimated mean of population 1"), 
            4)
        mu2 = round(st.number_input(
            label="μ₂", 
            min_value=-1e10, 
            max_value=1e10, 
            value=0.0, 
            step=1.0, 
            format="%.2f", 
            help="Estimated mean of population 2"), 
            4)
        s1 = round(st.number_input(
            label="s₁", 
            min_value=0.01, 
            max_value=1e10, 
            value=1.0, 
            step=1.0, 
            format="%.2f", 
            help="Estimated standard deviation of population 1"),
            4)
        s2 = round(st.number_input(
            label="s₂", 
            min_value=0.01, 
            max_value=1e10, 
            value=1.0, 
            step=1.0, 
            format="%.2f", 
            help="Estimated standard deviation of population 2"), 
            4)
        
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

    n1 = round(st.number_input(
        label="Group 1 Sample Size", 
        min_value=2, 
        value=50), 
        4)
    
    n2 = round(st.number_input(
        label="Group 2 Sample Size", 
        min_value=2, 
        value=50), 
        4)
    
    power_target = round(st.number_input(
        label="Desired Power", 
        min_value=0.01, 
        max_value=0.99, 
        value=0.8, 
        step=0.01,
        help="Statistical power. This is the probability of correctly rejecting the null hypothesis when it is false."), 
        4)

    effect_size = effect_size_two_sample_ttest(n1, n2, power_target, alpha, alternative)
    st.success(f"Effect Size: **{effect_size:.3f}**")