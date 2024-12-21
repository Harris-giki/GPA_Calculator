import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from grading_logic import GradingSystem, GPACalculator

def save_plot_to_bytes(fig):
    """Convert matplotlib figure to bytes for downloading"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def main():
    st.set_page_config(page_title="Advanced Grading System", layout="wide")
    st.title("Advanced Grading System")
    st.markdown("---")

    # Initialize session state for storing results
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'figures' not in st.session_state:
        st.session_state.figures = {}

    # Sidebar configuration
    st.sidebar.header("System Configuration")
    system_choice = st.sidebar.radio(
        "Choose Grading System Type:",
        ["Relative Grading", "Absolute Grading"]
    )

    # File upload
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        if system_choice == "Relative Grading":
            handle_relative_grading(uploaded_file)
        else:
            handle_absolute_grading(uploaded_file)

def handle_relative_grading(uploaded_file):
    grading_system = GradingSystem()
    
    try:
        # Load data
        grading_system.data = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded data for {len(grading_system.data)} students")

        # Credit hours input
        st.header("Credit Hours Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Enter credit hours for each subject:")
            credit_hours_set = True
            for subject in grading_system.subject_columns:
                subject_name = subject.replace('_score', '')
                hours = st.number_input(
                    f"Credit hours for {subject_name}",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                    key=f"rel_{subject}"
                )
                grading_system.credit_hours[subject] = hours

        with col2:
            st.write("Preview of loaded data:")
            st.dataframe(grading_system.data.head())

        # Grading method selection
        method = st.selectbox(
            "Choose grading method:",
            ["HEC", "Custom"]
        )

        custom_distribution = None
        if method == "Custom":
            st.subheader("Custom Grade Distribution")
            st.write("Enter percentage distribution for grades (should sum to 100)")
            
            custom_distribution = {}
            total = 0
            col1, col2 = st.columns(2)
            
            grades = ['A', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D']
            for i, grade in enumerate(grades):
                with col1 if i < len(grades)/2 else col2:
                    percentage = st.slider(
                        f"Percentage for {grade}",
                        0.0,
                        100.0 - total,
                        10.0,
                        0.1,
                        key=f"dist_{grade}"
                    )
                    total += percentage
                    custom_distribution[grade] = percentage / 100

            st.write(f"Total percentage: {total}%")
            if abs(total - 100) > 0.1:
                st.error("Total percentage must equal 100%")
                return

        if st.button("Process Grades"):
            with st.spinner("Processing grades..."):
                results = grading_system.process_grades(method, custom_distribution)
                st.session_state.results = results
                
                # Display results
                display_relative_results(results, grading_system)

def display_relative_results(results, grading_system):
    st.header("Results Analysis")
    
    # Overall CGPA Statistics
    st.subheader("Overall CGPA Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average CGPA", f"{results['cgpa'].mean():.2f}")
    col2.metric("Highest CGPA", f"{results['cgpa'].max():.2f}")
    col3.metric("Lowest CGPA", f"{results['cgpa'].min():.2f}")
    col4.metric("Standard Deviation", f"{results['cgpa'].std():.2f}")

    # CGPA Distribution Plot
    fig_cgpa = plt.figure(figsize=(10, 6))
    sns.histplot(data=results, x='cgpa', bins=20)
    plt.title('CGPA Distribution')
    st.pyplot(fig_cgpa)
    st.session_state.figures['cgpa_dist'] = fig_cgpa

    # Top Performers
    st.subheader("Top 5 Performers")
    top_5 = results.nlargest(5, 'cgpa')[['first_name', 'last_name', 'cgpa']]
    st.table(top_5)

    # Subject-wise Analysis
    st.subheader("Subject-wise Analysis")
    for subject in grading_system.subject_columns:
        subject_name = subject.replace('_score', '')
        with st.expander(f"{subject_name.upper()} Analysis"):
            col1, col2 = st.columns(2)
            
            # Score statistics
            scores = results[subject]
            with col1:
                st.write("Score Statistics:")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{scores.mean():.2f}",
                        f"{scores.median():.2f}",
                        f"{scores.std():.2f}",
                        f"{scores.min():.2f}",
                        f"{scores.max():.2f}"
                    ]
                })
                st.table(stats_df)
            
            # Grade distribution
            with col2:
                grade_col = f"{subject_name}_grade"
                grade_counts = results[grade_col].value_counts().sort_index()
                fig_grade = plt.figure(figsize=(8, 6))
                grade_counts.plot(kind='bar')
                plt.title(f'{subject_name} Grade Distribution')
                plt.tight_layout()
                st.pyplot(fig_grade)
                st.session_state.figures[f'{subject_name}_grade_dist'] = fig_grade

    # Download section
    st.header("Download Results")
    col1, col2 = st.columns(2)
    
    # Download results CSV
    with col1:
        st.download_button(
            "Download Results CSV",
            results.to_csv(index=False).encode('utf-8'),
            "grading_results.csv",
            "text/csv",
            key='download-csv'
        )
    
    # Download all plots
    with col2:
        for name, fig in st.session_state.figures.items():
            plot_bytes = save_plot_to_bytes(fig)
            st.download_button(
                f"Download {name.replace('_', ' ').title()} Plot",
                plot_bytes,
                f"{name}.png",
                "image/png",
                key=f'download-{name}'
            )

def handle_absolute_grading(uploaded_file):
    calculator = GPACalculator()
    
    try:
        # Load data
        calculator.df = pd.read_csv(uploaded_file)
        st.success(f"Loaded data for {len(calculator.df)} students")

        # Credit hours input
        st.header("Credit Hours Configuration")
        subjects = ['math', 'history', 'physics', 'chemistry', 'biology', 'english', 'geography']
        
        col1, col2 = st.columns(2)
        with col1:
            for subject in subjects:
                hours = st.number_input(
                    f"Credit hours for {subject}",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key=f"abs_{subject}"
                )
                calculator.subject_credits[subject] = hours
        
        with col2:
            st.write("Preview of loaded data:")
            st.dataframe(calculator.df.head())

        # Grading method selection
        method = st.radio(
            "Choose grading method:",
            ["HEC Standard", "Custom Thresholds"]
        )

        if st.button("Calculate Results"):
            with st.spinner("Processing..."):
                if method == "HEC Standard":
                    results = calculator.calculate_results(calculator.hec_thresholds)
                    stats = calculator.generate_statistics(results)
                    st.session_state.results = results
                    
                    st.header("Results (HEC Standard)")
                    display_absolute_results(results, stats)
                    
                else:
                    custom_thresholds = get_custom_thresholds_ui()
                    if custom_thresholds:
                        results = calculator.calculate_results(custom_thresholds)
                        stats = calculator.generate_statistics(results)
                        st.session_state.results = results
                        
                        st.header("Results (Custom Thresholds)")
                        display_absolute_results(results, stats)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def get_custom_thresholds_ui():
    """UI for custom threshold input"""
    st.subheader("Custom Grade Thresholds")
    thresholds = {}
    grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D']
    
    st.write("Enter score ranges and grade points for each grade:")
    
    for grade in grades:
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.number_input(f"{grade} - Min Score", 0, 100, key=f"min_{grade}")
        with col2:
            max_score = st.number_input(f"{grade} - Max Score", 0, 100, key=f"max_{grade}")
        with col3:
            points = st.number_input(f"{grade} - Grade Points", 0.0, 4.0, key=f"points_{grade}")
            
        thresholds[grade] = (min_score, max_score, points)
    
    # Add F grade automatically
    min_threshold = min(min_score for min_score, _, _ in thresholds.values())
    thresholds['F'] = (0, min_threshold - 0.01, 0.00)
    
    # Validate thresholds
    if st.button("Validate Thresholds"):
        if validate_thresholds(thresholds):
            st.success("Thresholds are valid!")
            return thresholds
        else:
            st.error("Invalid thresholds. Please check the ranges and points.")
            return None

def validate_thresholds(thresholds):
    """Validate that thresholds don't overlap and are properly ordered"""
    sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1][1], reverse=True)
    previous_min = None
    points_set = set()
    
    for grade, (min_score, max_score, points) in sorted_thresholds:
        if not (0 <= min_score <= max_score <= 100):
            return False
        if previous_min is not None and min_score >= previous_min:
            return False
        if points in points_set:
            return False
        points_set.add(points)
        previous_min = min_score
    
    return True

def display_absolute_results(results, stats):
    """Display results for absolute grading"""
    # GPA Statistics
    st.subheader("GPA Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean GPA", f"{stats['gpa_stats']['mean']:.2f}")
    col2.metric("Median GPA", f"{stats['gpa_stats']['median']:.2f}")
    col3.metric("Highest GPA", f"{stats['gpa_stats']['max']:.2f}")
    col4.metric("Lowest GPA", f"{stats['gpa_stats']['min']:.2f}")

    # GPA Distribution Plot
    fig_gpa = plt.figure(figsize=(10, 6))
    sns.histplot(data=results, x='gpa', bins=20)
    plt.title('GPA Distribution')
    st.pyplot(fig_gpa)
    st.session_state.figures['gpa_dist'] = fig_gpa

    # Subject Analysis
    st.subheader("Subject-wise Analysis")
    subjects = ['math', 'history', 'physics', 'chemistry', 'biology', 'english', 'geography']
    
    for subject in subjects:
        with st.expander(f"{subject.upper()} Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                stats_key = f"{subject}_score_stats"
                if stats_key in stats:
                    st.write("Score Statistics:")
                    stats_df = pd.DataFrame(stats[stats_key].items(), columns=['Metric', 'Value'])
                    st.table(stats_df)
            
            with col2:
                dist_key = f"{subject}_grade_distribution"
                if dist_key in stats:
                    grade_dist = pd.Series(stats[dist_key])
                    fig_grade = plt.figure(figsize=(8, 6))
                    grade_dist.plot(kind='bar')
                    plt.title(f'{subject} Grade Distribution')
                    plt.tight_layout()
                    st.pyplot(fig_grade)
                    st.session_state.figures[f'{subject}_grade_dist'] = fig_grade

    # Download section
    st.header("Download Results")
    col1, col2 = st.columns(2)
    
    # Download results CSV
    with col1:
        st.download_button(
            "Download Results CSV",
            results.to_csv(index=False).encode('utf-8'),
            "grading_results.csv",
            "text/csv",
            key='download-csv'
        )
    
    # Download all plots
    with col2:
        for name, fig in st.session_state.figures.items():
            plot_bytes = save_plot_to_bytes(fig)
            st.download_button(
                f"Download {name.replace('_', ' ').title()} Plot",
                plot_bytes,
                f"{name}.png",
                "image/png",
                key=f'download-{name}'
            )

if __name__ == "__main__":
    main() 
