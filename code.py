import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from scipy import stats
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=f'grading_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StreamlitGradingSystem:
    def __init__(self):
        self.data = None
        self.credit_hours = {}
        self.subject_columns = [
            'math_score', 'history_score', 'physics_score',
            'chemistry_score', 'biology_score', 'english_score',
            'geography_score'
        ]
        # HEC predefined grade thresholds for absolute grading
        self.hec_thresholds = {
            'A': (85, 100, 4.00),
            'A-': (80, 84, 3.66),
            'B+': (75, 79, 3.33),
            'B': (71, 74, 3.00),
            'B-': (68, 70, 2.66),
            'C+': (64, 67, 2.33),
            'C': (61, 63, 2.00),
            'C-': (58, 60, 1.66),
            'D+': (54, 57, 1.30),
            'D': (50, 53, 1.00),
            'F': (0, 49, 0.00)
        }

    def load_data(self, uploaded_file) -> None:
        """Load student data from uploaded CSV file."""
        try:
            self.data = pd.read_csv(uploaded_file)
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def set_credit_hours(self, credits_dict: Dict[str, float]) -> None:
        """Set credit hours from dictionary input"""
        self.credit_hours = credits_dict

    def calculate_statistics(self, scores: np.array) -> Dict:
        """Calculate descriptive statistics for a set of scores."""
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'skewness': stats.skew(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }

    def apply_hec_relative_grading(self, scores: np.array) -> List[str]:
        """Apply HEC relative grading based on standard deviations."""
        mean = np.mean(scores)
        std = np.std(scores)
        
        grades = []
        for score in scores:
            if score > mean + 2*std:
                grades.append('A*')
            elif score > mean + (3/2)*std:
                grades.append('A')
            elif score > mean + std:
                grades.append('A-')
            elif score > mean + std/2:
                grades.append('B+')
            elif score > mean - std/2:
                grades.append('B')
            elif score > mean - std:
                grades.append('B-')
            elif score > mean - (4/3)*std:
                grades.append('C+')
            elif score > mean - (5/3)*std:
                grades.append('C')
            elif score > mean - 2*std:
                grades.append('C-')
            else:
                grades.append('D')
        return grades

    def apply_absolute_grading(self, scores: np.array, thresholds: Dict) -> List[str]:
        """Apply absolute grading based on threshold values."""
        grades = []
        for score in scores:
            for grade, (min_score, max_score, _) in thresholds.items():
                if min_score <= score <= max_score:
                    grades.append(grade)
                    break
        return grades

    def calculate_gpa(self, grade: str) -> float:
        """Convert letter grade to GPA."""
        gpa_scale = {
            'A*': 4.00, 'A': 4.00, 'A-': 3.67,
            'B+': 3.33, 'B': 3.00, 'B-': 2.67,
            'C+': 2.33, 'C': 2.00, 'C-': 1.67,
            'D+': 1.33, 'D': 1.00, 'F': 0.00
        }
        return gpa_scale.get(grade, 0.0)

    def process_grades(self, grading_type: str, grading_method: str, 
                      custom_distribution: Dict[str, float] = None,
                      custom_thresholds: Dict[str, Tuple[int, int, float]] = None) -> pd.DataFrame:
        """Process grades using specified method."""
        results = pd.DataFrame()
        results[['first_name', 'last_name', 'email']] = self.data[['first_name', 'last_name', 'email']]
        
        total_credit_hours = sum(self.credit_hours.values())
        weighted_gpa = 0

        for subject in self.subject_columns:
            scores = self.data[subject].values
            
            if grading_type == 'Relative':
                if grading_method == 'HEC':
                    grades = self.apply_hec_relative_grading(scores)
                else:
                    grades = self.apply_custom_relative_grading(scores, custom_distribution)
            else:  # Absolute
                thresholds = self.hec_thresholds if grading_method == 'HEC' else custom_thresholds
                grades = self.apply_absolute_grading(scores, thresholds)
            
            results[f'{subject}_grade'] = grades
            subject_gpas = [self.calculate_gpa(grade) for grade in grades]
            results[f'{subject}_gpa'] = subject_gpas
            weighted_gpa += np.array(subject_gpas) * self.credit_hours[subject.replace('_score', '')]

        results['cgpa'] = weighted_gpa / total_credit_hours
        return results

    def visualize_results(self, results: pd.DataFrame) -> None:
        """Create and display visualizations."""
        # CGPA Distribution
        st.subheader("CGPA Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=results, x='cgpa', bins=20)
        plt.title("CGPA Distribution")
        st.pyplot(fig)

        # Subject-wise Score Distribution
        st.subheader("Subject-wise Score Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        score_data = [results[col] for col in self.subject_columns]
        plt.boxplot(score_data, labels=[col.replace('_score', '') for col in self.subject_columns])
        plt.xticks(rotation=45)
        plt.title("Score Distribution Across Subjects")
        st.pyplot(fig)

        # Grade Distribution Heatmap
        st.subheader("Grade Distribution Heatmap")
        grade_cols = [col for col in results.columns if col.endswith('_grade')]
        grade_dist = pd.DataFrame()
        for col in grade_cols:
            grade_dist[col.replace('_grade', '')] = results[col].value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(grade_dist, annot=True, fmt='g', cmap='YlGnBu')
        plt.title("Grade Distribution Across Subjects")
        st.pyplot(fig)

        # Correlation Matrix
        st.subheader("Subject Score Correlations")
        fig, ax = plt.subplots(figsize=(10, 8))
        score_correlation = results[[col for col in results.columns if col.endswith('_score')]].corr()
        sns.heatmap(score_correlation, annot=True, cmap='coolwarm', center=0)
        plt.title("Subject Score Correlations")
        st.pyplot(fig)

def main():
    st.set_page_config(page_title="Advanced Student Grading System", layout="wide")
    st.title("Advanced Student Grading System")

    # Initialize grading system
    grading_system = StreamlitGradingSystem()

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None and grading_system.load_data(uploaded_file):
        st.success("Data loaded successfully!")

        # Credit hours input
        st.sidebar.subheader("Credit Hours")
        credit_hours = {}
        for subject in ['math', 'history', 'physics', 'chemistry', 'biology', 'english', 'geography']:
            credit_hours[subject] = st.sidebar.number_input(
                f"{subject.capitalize()} credits",
                min_value=1.0,
                max_value=6.0,
                value=3.0,
                step=0.5
            )
        grading_system.set_credit_hours(credit_hours)

        # Grading type selection
        grading_type = st.sidebar.selectbox(
            "Select Grading Type",
            ["Relative", "Absolute"]
        )

        # Grading method selection
        grading_method = st.sidebar.selectbox(
            "Select Grading Method",
            ["HEC", "Custom"]
        )

        custom_distribution = None
        custom_thresholds = None

        if grading_method == "Custom":
            if grading_type == "Relative":
                st.sidebar.subheader("Custom Grade Distribution")
                custom_distribution = {}
                total = 0
                for grade in ['A', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D']:
                    value = st.sidebar.slider(
                        f"Percentage for {grade}",
                        0.0,
                        100.0 - total,
                        10.0,
                        1.0
                    )
                    custom_distribution[grade] = value / 100
                    total += value
            else:  # Absolute
                st.sidebar.subheader("Custom Grade Thresholds")
                custom_thresholds = {}
                for grade in ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D']:
                    col1, col2 = st.sidebar.columns(2)
                    min_score = col1.number_input(f"{grade} Min", 0, 100, key=f"min_{grade}")
                    max_score = col2.number_input(f"{grade} Max", 0, 100, key=f"max_{grade}")
                    points = st.sidebar.number_input(f"{grade} Points", 0.0, 4.0, key=f"points_{grade}")
                    custom_thresholds[grade] = (min_score, max_score, points)

        if st.sidebar.button("Process Grades"):
            results = grading_system.process_grades(
                grading_type,
                grading_method,
                custom_distribution,
                custom_thresholds
            )
            
            # Display results
            st.header("Results")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average CGPA", f"{results['cgpa'].mean():.2f}")
            with col2:
                st.metric("Highest CGPA", f"{results['cgpa'].max():.2f}")
            with col3:
                st.metric("Lowest CGPA", f"{results['cgpa'].min():.2f}")
            with col4:
                st.metric("CGPA Std Dev", f"{results['cgpa'].std():.2f}")

            # Visualizations
            grading_system.visualize_results(results)

            # Detailed results table
            st.subheader("Detailed Student Results")
            st.dataframe(results)

            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="grading_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
