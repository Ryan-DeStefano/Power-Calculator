import streamlit as st



def validate_effect_size_direction_power(effect_size, alternative):
    if alternative == 'larger' and effect_size < 0:
        return "**WARNING:** Effect size is negative but test type is 'larger'. Consider flipping the sign or changing the test direction."
    elif alternative == 'smaller' and effect_size > 0:
        return "**WARNING:** Effect size is positive but test type is 'smaller'. Consider flipping the sign or changing the test direction."
    return None

def input_sample_size(group_label, key, default=50):
    return round(
        st.number_input(
            label=group_label,
            min_value=2,
            value=default,
            key=key
        ),
        4
    )

def input_effect_size(key, default=0.5):
    return round(
        st.number_input(
            label="Effect Size (Cohen's d)",
            min_value=-2.0,
            max_value=2.0,
            value=default,
            step=0.01,
            format="%.2f",
            key=key,
            help="See top of page for explanation of Cohen's d"
        ),
        4
    )

def input_mean(group_label, key, default=0.0):
    return round(
        st.number_input(
            label=f"μ{group_label}",
            min_value=-1e10,
            max_value=1e10,
            value=default,
            step=1.0,
            format="%.2f",
            key=key,
            help=f"Estimated mean of population {group_label}"
        ),
        4
    )

def input_std(group_label, key, default=1.0):
    return round(
        st.number_input(
            label=f"s{group_label}",
            min_value=0.01,
            max_value=1e10,
            value=default,
            step=1.0,
            format="%.2f",
            key=key,
            help=f"Estimated standard deviation of population {group_label}"
        ),
        4
    )

def input_power(key, default=0.8):
    return round(
        st.number_input(
            label="Desired Power", 
            min_value=0.01, 
            max_value=0.99, 
            value=default, 
            step=0.01,
            key=key,
            help="Statistical power. This is the probability of correctly rejecting the null hypothesis when it is false."
        ), 
        4
    )

def input_sample_ratio(key, default=1.0):
    return round(
        st.number_input(
            label="Ratio of Group 1 Size to Group 2 Size",
            min_value=0.01, 
            max_value=1000.0, 
            value=default, 
            step=0.1,
            key=key, 
            help="The ratio n₁/n₂ of the two sample sizes. Use 1.0 for equal group sizes. For example, 2.0 means Group 1 has twice as many samples as Group 2."
        ), 
        4
    ) 