import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from grading_logic import GradingSystem, GPACalculator
from facedetection import FaceDetector, process_student_image, train_model
import os
from datetime import datetime
import io
import base64
import scipy.stats as stats

# Set page config
st.set_page_config(
    page_title="Academic Grading System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'statistics' not in st.session_state:
    st.session_state.statistics = None

def main():
    st.title("Academic Grading System 📚")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a method:",
        ["Home", "Automated Student Detection", "Absolute Grading", "Relative Grading"]
    )

    if page == "Home":
        show_home_page()
    elif page == "Automated Student Detection":
        show_automated_detection()
    elif page == "Absolute Grading":
        show_absolute_grading()
    else:
        show_relative_grading()

def show_home_page():
    st.markdown("""
    ## Welcome to the Academic Grading System
    
    This system provides comprehensive grading functionality with two main methods:
    
    ### 1. Absolute Grading
    - Use predefined HEC thresholds
    - Define custom grade thresholds
    - Detailed statistical analysis
    - Comparative visualizations
    
    ### 2. Relative Grading
    - HEC relative grading system
    - Custom grade distribution
    - Statistical analysis with curve fitting
    - Comprehensive reporting
    
    ### Getting Started:
    1. Select your preferred grading method from the sidebar
    2. Upload your student data CSV file
    3. Configure grading parameters
    4. View results and download reports
    
    ### Required Data Format:
    Your CSV file should include these columns:
    - first_name, last_name, email
    - Subject scores (math_score, history_score, etc.)
    - Additional student information (gender, part_time_job, etc.)
    """)

def show_absolute_grading():
    st.header("Absolute Grading System")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        calculator = GPACalculator()
        
        try:
            # Load data
            calculator.load_data(uploaded_file)
            st.success("Data loaded successfully!")
            
            # Credit hours input
            st.subheader("Enter Credit Hours")
            subjects = ['math', 'history', 'physics', 'chemistry', 'biology', 'english', 'geography']
            
            col1, col2 = st.columns(2)
            credit_hours = {}
            
            with col1:
                for subject in subjects[:4]:
                    credit_hours[subject] = st.number_input(
                        f"Credit hours for {subject}",
                        min_value=1,
                        max_value=6,
                        value=3
                    )
            
            with col2:
                for subject in subjects[4:]:
                    credit_hours[subject] = st.number_input(
                        f"Credit hours for {subject}",
                        min_value=1,
                        max_value=6,
                        value=3
                    )
            
            calculator.subject_credits = credit_hours
            
            # Grading method selection
            grading_method = st.radio(
                "Select grading method:",
                ["HEC Predefined Thresholds", "Custom Thresholds"]
            )
            
            if grading_method == "Custom Thresholds":
                st.subheader("Define Custom Grade Thresholds")
                
                # Initialize default thresholds
                if 'thresholds' not in st.session_state:
                    st.session_state.thresholds = {
                        'A': (90, 100),
                        'A-': (85, 89),
                        'B+': (80, 84),
                        'B': (75, 79),
                        'B-': (70, 74),
                        'C+': (65, 69),
                        'C': (60, 64),
                        'C-': (55, 59),
                        'D': (50, 54)
                    }
                
                custom_thresholds = {}
                grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D']
                
                st.info("Adjust grade thresholds using the sliders. Ranges will automatically adjust to prevent overlapping.")
                
                # Create sliders for each grade
                for i, grade in enumerate(grades):
                    st.write(f"### Grade {grade}")
                    
                    # Get current threshold values
                    current_min, current_max = st.session_state.thresholds[grade]
                    
                    # Create range slider
                    min_val, max_val = st.slider(
                        f"Score range for {grade}",
                        min_value=0.0,
                        max_value=100.0,
                        value=(float(current_min), float(current_max)),
                        step=0.5,
                        key=f"range_{grade}"
                    )
                    
                    # Add grade points input
                    points = st.number_input(
                        f"Grade points for {grade}",
                        min_value=0.0,
                        max_value=4.0,
                        value=4.0 - (i * 0.33),  # Decreasing grade points
                        step=0.01,
                        key=f"points_{grade}"
                    )
                    
                    custom_thresholds[grade] = (min_val, max_val, points)
                    st.write(f"Range: {min_val:.1f} - {max_val:.1f}, Points: {points:.2f}")
                    st.write("---")
                
                if st.button("Apply Custom Thresholds"):
                    # Validate thresholds
                    valid = True
                    for i in range(len(grades)-1):
                        if custom_thresholds[grades[i]][0] <= custom_thresholds[grades[i+1]][1]:
                            valid = False
                            st.error(f"Error: {grades[i]} threshold overlaps with {grades[i+1]}")
                    
                    if valid:
                        try:
                            # Calculate results using both methods
                            hec_results = calculator.calculate_results(calculator.hec_thresholds)
                            custom_results = calculator.calculate_results(custom_thresholds)
                            
                            # Display comparative results
                            display_comparative_results(hec_results, custom_results, calculator)
                            
                        except Exception as e:
                            st.error(f"Error calculating grades: {str(e)}")
            
            else:  # HEC Thresholds
                if st.button("Calculate Grades"):
                    try:
                        results = calculator.calculate_results(calculator.hec_thresholds)
                        display_hec_results(results, calculator)
                    except Exception as e:
                        st.error(f"Error calculating grades: {str(e)}")
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
def show_relative_grading():
    st.header("Relative Grading System")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        grading_system = GradingSystem()
        
        try:
            # Load data
            grading_system.load_data(uploaded_file)
            st.success("Data loaded successfully!")
            
            # Grading method selection
            grading_method = st.radio(
                "Select grading method:",
                ["HEC Relative Grading", "Custom Standard Deviation Thresholds"]
            )
            
            if grading_method == "Custom Standard Deviation Thresholds":
                st.subheader("Define Grade Boundaries using Standard Deviations")
                st.info("""
                Define grade boundaries using standard deviations from the mean.
                - Positive values are above the mean (μ)
                - Negative values are below the mean
                - Mean (μ) is at 0 standard deviations
                """)
                
                # Dictionary to store std thresholds
                custom_std_thresholds = {}
                
                # Create columns for better organization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Upper Grades")
                    # A grade (above certain σ)
                    a_threshold = st.number_input(
                        "A grade:",
                        min_value=0.0,
                        max_value=3.0,
                        value=2.0,
                        step=0.1,
                        help="Scores above this many standard deviations from mean will get A"
                    )
                    custom_std_thresholds['A'] = (a_threshold, float('inf'))
                    
                    # B+ grade
                    bplus_upper = a_threshold  # Upper bound is A's lower bound
                    bplus_lower = st.number_input(
                        "B+ grade:",
                        min_value=0.0,
                        max_value=bplus_upper,
                        value=min(1.5, bplus_upper),
                        step=0.1
                    )
                    custom_std_thresholds['B+'] = (bplus_lower, bplus_upper)
                    
                    # B grade
                    b_upper = bplus_lower
                    b_lower = st.number_input(
                        "B grade:",
                        min_value=0.0,
                        max_value=b_upper,
                        value=min(1.0, b_upper),
                        step=0.1
                    )
                    custom_std_thresholds['B'] = (b_lower, b_upper)
                    
                    # B- grade
                    bminus_upper = b_lower
                    bminus_lower = st.number_input(
                        "B- grade:",
                        min_value=0.0,
                        max_value=bminus_upper,
                        value=min(0.5, bminus_upper),
                        step=0.1
                    )
                    custom_std_thresholds['B-'] = (bminus_lower, bminus_upper)
                
                with col2:
                    st.markdown("### Lower Grades")
                    # C+ grade
                    cplus_upper = bminus_lower
                    cplus_lower = st.number_input(
                        "C+ grade",
                        min_value=-3.0,
                        max_value=cplus_upper,
                        value=0.0,
                        step=0.1
                    )
                    custom_std_thresholds['C+'] = (cplus_lower, cplus_upper)
                    
                    # C grade
                    c_upper = cplus_lower
                    c_lower = st.number_input(
                        "C grade",
                        min_value=-3.0,
                        max_value=c_upper,
                        value=-0.5,
                        step=0.1
                    )
                    custom_std_thresholds['C'] = (c_lower, c_upper)
                    
                    # C- grade
                    cminus_upper = c_lower
                    cminus_lower = st.number_input(
                        "C- grade",
                        min_value=-3.0,
                        max_value=cminus_upper,
                        value=-1.0,
                        step=0.1
                    )
                    custom_std_thresholds['C-'] = (cminus_lower, cminus_upper)
                    
                    # D grade (below certain σ)
                    custom_std_thresholds['D'] = (float('-inf'), cminus_lower)
                
                # Display current grade boundaries
                st.markdown("### Current Grade Boundaries")
                boundaries_df = pd.DataFrame([
                    ["A", f"Above {a_threshold}σ"],
                    ["B+", f"Between {bplus_lower}σ and {bplus_upper}σ"],
                    ["B", f"Between {b_lower}σ and {b_upper}σ"],
                    ["B-", f"Between {bminus_lower}σ and {bminus_upper}σ"],
                    ["C+", f"Between {cplus_lower}σ and {cplus_upper}σ"],
                    ["C", f"Between {c_lower}σ and {c_upper}σ"],
                    ["C-", f"Between {cminus_lower}σ and {cminus_upper}σ"],
                    ["D", f"Below {cminus_lower}σ"]
                ], columns=["Grade", "Boundary"])
                st.table(boundaries_df)
                
                if st.button("Apply Custom Thresholds"):
                    try:
                        # Create the thresholds dictionary
                        std_thresholds = {
                            'A': (custom_std_thresholds['A'][0], float('inf')),
                            'B+': (custom_std_thresholds['B+'][0], custom_std_thresholds['A'][0]),
                            'B': (custom_std_thresholds['B'][0], custom_std_thresholds['B+'][0]),
                            'B-': (custom_std_thresholds['B-'][0], custom_std_thresholds['B'][0]),
                            'C+': (custom_std_thresholds['C+'][0], custom_std_thresholds['B-'][0]),
                            'C': (custom_std_thresholds['C'][0], custom_std_thresholds['C+'][0]),
                            'C-': (custom_std_thresholds['C-'][0], custom_std_thresholds['C'][0]),
                            'D': (float('-inf'), custom_std_thresholds['C-'][0])
                        }
                        
                        # Calculate results
                        results = grading_system.process_grades("CUSTOM", std_thresholds)
                        display_relative_results(results, grading_system, "Custom")
                        
                    except Exception as e:
                        st.error(f"Error calculating grades: {str(e)}")
            
            else:  # HEC Relative Grading
                if st.button("Calculate Grades"):
                    results = grading_system.process_grades("HEC")
                    display_relative_results(results, grading_system, "HEC")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def display_relative_comparative_results(hec_results, custom_results, grading_system):
    st.subheader("Comparative Analysis: HEC vs Custom Distribution")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Grade Distribution", "GPA Analysis", "Statistical Summary"])
    
    with tab1:
        st.subheader("Subject-wise Grade Distribution Comparison")
        
        # Get subject columns
        subject_cols = [col for col in hec_results.columns if col.endswith('_grade')]
        
        for subject in subject_cols:
            subject_name = subject.replace('_grade', '').title()
            
            # Create two-column comparison
            st.write(f"\n### {subject_name}")
            
            # Create comparison plot
            fig = go.Figure()
            
            # Get all possible grades from both distributions
            all_grades = sorted(list(set(
                list(hec_results[subject].unique()) + 
                list(custom_results[subject].unique())
            )))
            
            # Calculate distributions with consistent grade scales
            hec_dist = hec_results[subject].value_counts().reindex(all_grades).fillna(0)
            custom_dist = custom_results[subject].value_counts().reindex(all_grades).fillna(0)
            
            # Add HEC distribution
            fig.add_trace(go.Bar(
                x=all_grades,
                y=hec_dist.values,
                name="HEC",
                marker_color='blue',
                opacity=0.6
            ))
            
            # Add Custom distribution
            fig.add_trace(go.Bar(
                x=all_grades,
                y=custom_dist.values,
                name="Custom",
                marker_color='red',
                opacity=0.6
            ))
            
            fig.update_layout(
                barmode='group',
                title=f"{subject_name} Grade Distribution Comparison",
                xaxis_title="Grade",
                yaxis_title="Number of Students",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add numerical comparison
            comparison_df = pd.DataFrame({
                'Grade': all_grades,
                'HEC Count': hec_dist.values,
                'HEC %': (hec_dist.values / len(hec_results) * 100).round(2),
                'Custom Count': custom_dist.values,
                'Custom %': (custom_dist.values / len(custom_results) * 100).round(2)
            })
            st.write(comparison_df)
            st.write("---")
    
    with tab2:
        st.subheader("GPA Distribution Comparison")
        
        # Create GPA comparison plot
        fig = go.Figure()
        
        # Add HEC GPA distribution
        fig.add_trace(go.Histogram(
            x=hec_results['cgpa'] if 'cgpa' in hec_results.columns else hec_results['gpa'],
            name="HEC GPA",
            opacity=0.6,
            nbinsx=20,
            marker_color='blue'
        ))
        
        # Add Custom GPA distribution
        fig.add_trace(go.Histogram(
            x=custom_results['cgpa'] if 'cgpa' in custom_results.columns else custom_results['gpa'],
            name="Custom GPA",
            opacity=0.6,
            nbinsx=20,
            marker_color='red'
        ))
        
        fig.update_layout(
            barmode='overlay',
            title="GPA Distribution Comparison",
            xaxis_title="CGPA",
            yaxis_title="Number of Students",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add GPA statistics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("HEC GPA Statistics")
            gpa_col = 'cgpa' if 'cgpa' in hec_results.columns else 'gpa'
            st.write(hec_results[gpa_col].describe().round(2))
            
        with col2:
            st.write("Custom GPA Statistics")
            gpa_col = 'cgpa' if 'cgpa' in custom_results.columns else 'gpa'
            st.write(custom_results[gpa_col].describe().round(2))
    
    with tab3:
        st.subheader("Statistical Summary")
        
        # Create summary statistics for each subject
        for subject in subject_cols:
            subject_name = subject.replace('_grade', '').title()
            score_col = subject.replace('grade', 'score')
            
            if score_col in hec_results.columns and score_col in custom_results.columns:
                st.write(f"\n### {subject_name} Statistics")
                
                summary_stats = pd.DataFrame({
                    'HEC Mean': [hec_results[score_col].mean()],
                    'HEC Std': [hec_results[score_col].std()],
                    'Custom Mean': [custom_results[score_col].mean()],
                    'Custom Std': [custom_results[score_col].std()]
                }).round(2)
                
                st.write(summary_stats)
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        download_results(hec_results, "hec_relative_results.csv")
    with col2:
        download_results(custom_results, "custom_relative_results.csv")
def display_comparative_results(hec_results, custom_results, calculator):
    st.subheader("Comparative Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Grade Distribution", "GPA Analysis", "Statistical Summary"])
    
    with tab1:
        st.subheader("Subject-wise Grade Distribution")
        
        # Get subject columns
        subject_cols = [col for col in hec_results.columns if col.endswith('_grade')]
        
        # Create subject-wise comparisons
        for subject in subject_cols:
            subject_name = subject.replace('_grade', '').title()
            
            # Create two-column comparison
            st.write(f"\n### {subject_name}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("HEC Grading")
                hec_dist = hec_results[subject].value_counts().sort_index()
                fig = px.bar(
                    x=hec_dist.index,
                    y=hec_dist.values,
                    title=f"HEC Grade Distribution - {subject_name}",
                    labels={'x': 'Grade', 'y': 'Number of Students'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.write("Custom Grading")
                custom_dist = custom_results[subject].value_counts().sort_index()
                fig = px.bar(
                    x=custom_dist.index,
                    y=custom_dist.values,
                    title=f"Custom Grade Distribution - {subject_name}",
                    labels={'x': 'Grade', 'y': 'Number of Students'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=hec_results['gpa'], name="HEC GPA", opacity=0.7))
        fig.add_trace(go.Histogram(x=custom_results['gpa'], name="Custom GPA", opacity=0.7))
        fig.update_layout(
            title="GPA Distribution Comparison",
            barmode='overlay',
            xaxis_title="GPA",
            yaxis_title="Number of Students"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("HEC Grading Statistics")
            stats_hec = hec_results['gpa'].describe()
            st.write(pd.DataFrame({
                'Statistic': stats_hec.index,
                'Value': stats_hec.values.round(2)
            }))
            
        with col2:
            st.write("Custom Grading Statistics")
            stats_custom = custom_results['gpa'].describe()
            st.write(pd.DataFrame({
                'Statistic': stats_custom.index,
                'Value': stats_custom.values.round(2)
            }))
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        download_results(hec_results, "hec_results.csv")
    with col2:
        download_results(custom_results, "custom_results.csv")

def display_hec_results(results, calculator):
    st.subheader("HEC Grading Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Grade Distribution", "GPA Analysis", "Statistical Summary"])
    
    with tab1:
        st.subheader("Subject-wise Grade Distribution")
        
        # Get subject columns
        subject_cols = [col for col in results.columns if col.endswith('_grade')]
        
        # Create subject-wise visualizations
        for subject in subject_cols:
            subject_name = subject.replace('_grade', '').title()
            score_col = subject.replace('grade', 'score')  # Get corresponding score column
            grade_dist = results[subject].value_counts().sort_index()
            
            # Create two columns for each subject
            col1, col2 = st.columns(2)
            
            with col1:
                # Grade distribution visualization
                fig = px.bar(
                    x=grade_dist.index,
                    y=grade_dist.values,
                    title=f"{subject_name} Grade Distribution",
                    labels={'x': 'Grade', 'y': 'Number of Students'}
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title="Grade",
                    yaxis_title="Number of Students"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistical summary for the subject
                st.write(f"### {subject_name} Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{results[score_col].mean():.2f}",
                        f"{results[score_col].median():.2f}",
                        f"{results[score_col].std():.2f}",
                        f"{results[score_col].min():.2f}",
                        f"{results[score_col].max():.2f}"
                    ]
                })
                st.table(stats_df)
            
            # Grade distribution table below the visualizations
            st.write(f"Grade Distribution for {subject_name}:")
            summary_df = pd.DataFrame({
                'Grade': grade_dist.index,
                'Number of Students': grade_dist.values,
                'Percentage': (grade_dist.values / len(results) * 100).round(2)
            })
            st.write(summary_df)
            
            # Add score distribution histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=results[score_col],
                nbinsx=20,
                name="Score Distribution"
            ))
            fig.update_layout(
                title=f"{subject_name} Score Distribution",
                xaxis_title="Score",
                yaxis_title="Frequency",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("---")
    
    with tab2:
        st.subheader("GPA Analysis")
        
        # GPA Distribution
        fig = px.histogram(
            results,
            x='gpa',
            title="GPA Distribution",
            labels={'gpa': 'GPA', 'count': 'Number of Students'},
            nbins=20
        )
        fig.update_layout(
            xaxis_title="GPA",
            yaxis_title="Number of Students"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # GPA Statistics
        gpa_stats = pd.DataFrame({
            'Metric': ['Mean GPA', 'Median GPA', 'Std Dev', 'Minimum GPA', 'Maximum GPA'],
            'Value': [
                f"{results['gpa'].mean():.2f}",
                f"{results['gpa'].median():.2f}",
                f"{results['gpa'].std():.2f}",
                f"{results['gpa'].min():.2f}",
                f"{results['gpa'].max():.2f}"
            ]
        })
        st.table(gpa_stats)
    
    with tab3:
        st.subheader("Statistical Summary")
        
        # Create a comprehensive statistical summary for all subjects
        for subject in subject_cols:
            subject_name = subject.replace('_grade', '').title()
            score_col = subject.replace('grade', 'score')
            
            st.write(f"\n### {subject_name}")
            
            # Create detailed statistics
            detailed_stats = pd.DataFrame({
                'Metric': [
                    'Mean Score', 'Median Score', 'Standard Deviation',
                    'Minimum Score', 'Maximum Score', 'Skewness',
                    'Kurtosis', '25th Percentile', '75th Percentile'
                ],
                'Value': [
                    f"{results[score_col].mean():.2f}",
                    f"{results[score_col].median():.2f}",
                    f"{results[score_col].std():.2f}",
                    f"{results[score_col].min():.2f}",
                    f"{results[score_col].max():.2f}",
                    f"{results[score_col].skew():.2f}",
                    f"{results[score_col].kurtosis():.2f}",
                    f"{results[score_col].quantile(0.25):.2f}",
                    f"{results[score_col].quantile(0.75):.2f}"
                ]
            })
            st.table(detailed_stats)
            
            # Add box plot
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=results[score_col],
                name=subject_name,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            fig.update_layout(
                title=f"{subject_name} Score Distribution (Box Plot)",
                yaxis_title="Score",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("---")
    
    # Download results
    download_results(results, "hec_results.csv")
def display_relative_results(results, grading_system, method):
    st.subheader(f"{method} Relative Grading Results")
    
    # Add download button at the top
    st.write("### Download Results")
    if method == "Custom":
        download_results(results, "custom_relative_results.csv")
    else:  # HEC method
        download_results(results, "hec_relative_results.csv")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Grade Distribution", "GPA Analysis", "Statistical Summary", "Normal Distribution Analysis"])
    
    with tab1:
        # For custom grading, show side-by-side comparison with HEC in the Grade Distribution tab only
        if method == "Custom":
            st.subheader("HEC vs Custom Grading Comparison")
            
            # Calculate HEC results for comparison
            hec_results = grading_system.process_grades("HEC")
            
            # Add download buttons for both methods
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Download HEC Results")
                download_results(hec_results, "hec_relative_results.csv")
            with col2:
                st.write("### Download Custom Results")
                download_results(results, "custom_relative_results.csv")
            
            for subject in grading_system.subject_columns:
                subject_name = subject.replace('_score', '').title()
                grade_col = subject.replace('_score', '_grade')
                
                st.write(f"## {subject_name}")
                
                # Create summary tables
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### HEC Grading")
                    hec_dist = hec_results[grade_col].value_counts().sort_index()
                    # Create summary DataFrame
                    hec_summary = pd.DataFrame({
                        'Grade': hec_dist.index,
                        'Number of Students': hec_dist.values,
                        'Percentage': (hec_dist.values / len(hec_results) * 100).round(2)
                    })
                    hec_summary['Percentage'] = hec_summary['Percentage'].apply(lambda x: f"{x}%")
                    st.table(hec_summary)
                    
                    # Display bar chart
                    fig = px.bar(
                        x=hec_dist.index,
                        y=hec_dist.values,
                        title=f"HEC Grade Distribution - {subject_name}",
                        labels={'x': 'Grade', 'y': 'Number of Students'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("### Custom Grading")
                    custom_dist = results[grade_col].value_counts().sort_index()
                    # Create summary DataFrame
                    custom_summary = pd.DataFrame({
                        'Grade': custom_dist.index,
                        'Number of Students': custom_dist.values,
                        'Percentage': (custom_dist.values / len(results) * 100).round(2)
                    })
                    custom_summary['Percentage'] = custom_summary['Percentage'].apply(lambda x: f"{x}%")
                    st.table(custom_summary)
                    
                    # Display bar chart
                    fig = px.bar(
                        x=custom_dist.index,
                        y=custom_dist.values,
                        title=f"Custom Grade Distribution - {subject_name}",
                        labels={'x': 'Grade', 'y': 'Number of Students'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.write("---")
        else:
            # For HEC-only view, show single distribution with table
            for subject in grading_system.subject_columns:
                subject_name = subject.replace('_score', '').title()
                grade_col = subject.replace('_score', '_grade')
                
                st.write(f"## {subject_name}")
                
                # Create distribution
                grade_dist = results[grade_col].value_counts().sort_index()
                
                # Create summary DataFrame
                summary_df = pd.DataFrame({
                    'Grade': grade_dist.index,
                    'Number of Students': grade_dist.values,
                    'Percentage': (grade_dist.values / len(results) * 100).round(2)
                })
                summary_df['Percentage'] = summary_df['Percentage'].apply(lambda x: f"{x}%")
                
                # Display summary table
                st.table(summary_df)
                
                # Display bar chart
                fig = px.bar(
                    x=grade_dist.index,
                    y=grade_dist.values,
                    title=f"Grade Distribution - {subject_name}",
                    labels={'x': 'Grade', 'y': 'Number of Students'}
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write("---")
    
    with tab2:
        st.subheader("GPA Distribution")
        if method == "Custom":
            # Create comparative GPA histogram for HEC vs Custom
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=hec_results['gpa'],
                name="HEC GPA",
                opacity=0.7,
                nbinsx=20
            ))
            fig.add_trace(go.Histogram(
                x=results['gpa'],
                name="Custom GPA",
                opacity=0.7,
                nbinsx=20
            ))
            
            fig.update_layout(
                title="GPA Distribution Comparison",
                barmode='overlay',
                xaxis_title="GPA",
                yaxis_title="Number of Students"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics for both methods
            col1, col2 = st.columns(2)
            with col1:
                st.write("HEC GPA Statistics")
                st.write(hec_results['gpa'].describe().round(2))
            with col2:
                st.write("Custom GPA Statistics")
                st.write(results['gpa'].describe().round(2))
        else:
            # Original single histogram for HEC method
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=results['gpa'],
                nbinsx=20,
                marker_color='blue',
                opacity=0.6
            ))
            
            fig.update_layout(
                title="GPA Distribution",
                xaxis_title="GPA",
                yaxis_title="Number of Students",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.write("GPA Statistics")
            st.write(results['gpa'].describe().round(2))
    
    with tab3:
        st.subheader("Statistical Summary")
        for subject in grading_system.subject_columns:
            subject_name = subject.replace('_score', '').title()
            
            st.write(f"\n### {subject_name} Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Standard Deviation', 'Minimum', 'Maximum'],
                'Value': [
                    f"{results[subject].mean():.2f}",
                    f"{results[subject].std():.2f}",
                    f"{results[subject].min():.2f}",
                    f"{results[subject].max():.2f}"
                ]
            })
            st.write(stats_df)
    
    with tab4:
        st.subheader("Normal Distribution Analysis")
        
        # Overall grades normal distribution
        st.write("### Overall Grade Distribution")
        
        fig = make_subplots(rows=1, cols=2)
        
        # Calculate overall grades
        all_grades = []
        for subject in grading_system.subject_columns:
            all_grades.extend(results[subject].values)
        
        # Add histogram with normal curve
        hist_values, bin_edges = np.histogram(all_grades, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Fit normal distribution
        mu, std = stats.norm.fit(all_grades)
        x = np.linspace(min(all_grades), max(all_grades), 100)
        normal_curve = stats.norm.pdf(x, mu, std)
        
        # Plot histogram
        fig.add_trace(
            go.Histogram(x=all_grades, name="Actual Distribution", 
                        histnorm='probability density'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=normal_curve, name="Normal Curve",
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Q-Q plot
        fig.add_trace(
            go.Scatter(x=stats.norm.ppf(np.linspace(0.01, 0.99, len(all_grades))),
            y=np.sort(all_grades),
            mode='markers',
            name='Q-Q Plot'),
            row=1, col=2
        )
        
        # Add reference line
        qq_line = np.linspace(min(all_grades), max(all_grades))
        fig.add_trace(
            go.Scatter(x=qq_line, y=qq_line,
                      mode='lines',
                      name='Reference Line',
                      line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Overall Grade Distribution Analysis",
            showlegend=True,
            height=400
        )
        fig.update_xaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Subject-wise normal distribution analysis
        st.write("### Subject-wise Distribution Analysis")
        
        for subject in grading_system.subject_columns:
            subject_name = subject.replace('_score', '').title()
            st.write(f"\n#### {subject_name}")
            
            fig = make_subplots(rows=1, cols=2)
            
            scores = results[subject].values
            hist_values, bin_edges = np.histogram(scores, bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Fit normal distribution
            mu, std = stats.norm.fit(scores)
            x = np.linspace(min(scores), max(scores), 100)
            normal_curve = stats.norm.pdf(x, mu, std)
            
            # Plot histogram
            fig.add_trace(
                go.Histogram(x=scores, name="Actual Distribution",
                            histnorm='probability density'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=x, y=normal_curve, name="Normal Curve",
                          line=dict(color='red')),
                row=1, col=1
            )
            
            # Q-Q plot
            fig.add_trace(
                go.Scatter(x=stats.norm.ppf(np.linspace(0.01, 0.99, len(scores))),
                y=np.sort(scores),
                mode='markers',
                name='Q-Q Plot'),
                row=1, col=2
            )
            
            # Add reference line
            qq_line = np.linspace(min(scores), max(scores))
            fig.add_trace(
                go.Scatter(x=qq_line, y=qq_line,
                          mode='lines',
                          name='Reference Line',
                          line=dict(color='red', dash='dash')),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"{subject_name} Distribution Analysis",
                showlegend=True,
                height=400
            )
            fig.update_xaxes(title_text="Score", row=1, col=1)
            fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
            fig.update_yaxes(title_text="Density", row=1, col=1)
            fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add statistical test results
            _, p_value = stats.normaltest(scores)
            st.write(f"Normality Test p-value: {p_value:.4f}")
            if p_value < 0.05:
                st.write("The distribution significantly differs from normal (p < 0.05)")
            else:
                st.write("The distribution appears to be normal (p >= 0.05)")
            st.write("---")

def download_results(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)

def show_automated_detection():
    st.header("Automated Student Detection")
    
    # Check if model exists
    if not os.path.exists('face_recognition_model.pth'):
        st.warning("Model not trained yet. Please train the model first.")
    
    # Add option to train model
    if st.button("Train Model"):
        data_dir = "./data"  # Path to your data folder
        if not os.path.exists('data/train'):
            st.error("Training data not found. Please create a 'data/train' folder with student images.")
            return
            
        with st.spinner("Training model..."):
            try:
                train_model(data_dir)  # Pass the data_dir parameter
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    uploaded_image = st.file_uploader("Upload student image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        # Save the uploaded image temporarily
        temp_path = "temp_image." + uploaded_image.name.split('.')[-1]
        with open(temp_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Grading type selection
        grading_type = st.radio(
            "Select Grading Method:",
            ["Absolute", "Relative"],
            key="grading_type"
        )
        
        if st.button("Calculate GPA"):
            try:
                # Process the image and get results
                result = process_student_image(temp_path, grading_type.lower())
                
                if isinstance(result, dict):
                    # Create a success message box
                    st.success(f"Student Successfully Identified! (Confidence: {result['confidence']:.2%})")
                    
                    # Display results in a nice format
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Student Information")
                        st.write(f"**ID:** {result['student_id']}")
                        st.write(f"**Name:** {result['name']}")
                    
                    with col2:
                        st.markdown("### Academic Results")
                        st.write(f"**Grading Type:** {result['grading_type'].title()}")
                        st.write(f"**GPA:** {result['gpa']:.2f}")
                    
                else:
                    st.error(result)  # Display error message
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main() 