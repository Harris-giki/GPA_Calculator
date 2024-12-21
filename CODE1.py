import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=f'grading_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GradingSystem:
    def __init__(self):
        self.data = None
        self.credit_hours = {}
        self.subject_columns = [
            'math_score', 'history_score', 'physics_score',
            'chemistry_score', 'biology_score', 'english_score',
            'geography_score'
        ]

    def process_grades(self, scores, method='HEC', custom_distribution=None):
        if method == 'HEC':
            return self.apply_hec_relative_grading(scores)
        else:
            if not custom_distribution:
                raise ValueError("Custom distribution required for custom grading method")
            return self.apply_custom_relative_grading(scores, custom_distribution)

    def apply_hec_relative_grading(self, scores):
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

    def apply_custom_relative_grading(self, scores, distribution):
        sorted_scores = np.sort(scores)[::-1]
        total_students = len(scores)
        grades = [''] * total_students
        
        current_percentile = 0
        score_to_grade = {}
        
        for grade, percentage in distribution.items():
            threshold_index = int((current_percentile + percentage) * total_students) - 1
            if threshold_index >= 0 and threshold_index < total_students:
                threshold_score = sorted_scores[threshold_index]
                score_to_grade[threshold_score] = grade
            current_percentile += percentage
        
        for i, score in enumerate(scores):
            for threshold_score, grade in score_to_grade.items():
                if score >= threshold_score:
                    grades[i] = grade
                    break
            if not grades[i]:
                grades[i] = list(distribution.keys())[-1]
                
        return grades

def main():
    st.set_page_config(page_title="Student Grading System", layout="wide")
    
    st.title("Student Grading System")
    st.write("Upload your student data and configure grading parameters")
    
    # Initialize session state
    if 'grading_system' not in st.session_state:
        st.session_state.grading_system = GradingSystem()
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.grading_system.data = data
            st.success("Data loaded successfully!")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Credit hours input
            st.subheader("Credit Hours Configuration")
            col1, col2 = st.columns(2)
            
            credit_hours = {}
            subjects = [col.replace('_score', '') for col in st.session_state.grading_system.subject_columns]
            
            for i, subject in enumerate(subjects):
                with col1 if i < len(subjects)/2 else col2:
                    credit_hours[f"{subject}_score"] = st.number_input(
                        f"{subject.capitalize()} Credits",
                        min_value=1,
                        max_value=6,
                        value=3
                    )
            
            # Grading method selection
            st.subheader("Grading Method")
            grading_method = st.radio(
                "Select grading method:",
                ("HEC Relative Grading", "Custom Relative Grading")
            )
            
            custom_distribution = None
            if grading_method == "Custom Relative Grading":
                st.subheader("Custom Grade Distribution")
                st.write("Enter percentage for each grade (should sum to 100)")
                
                custom_distribution = {}
                col1, col2 = st.columns(2)
                
                with col1:
                    custom_distribution['A'] = st.slider('A Grade %', 0, 100, 10)
                    custom_distribution['B+'] = st.slider('B+ Grade %', 0, 100, 15)
                    custom_distribution['B'] = st.slider('B Grade %', 0, 100, 20)
                    custom_distribution['B-'] = st.slider('B- Grade %', 0, 100, 15)
                
                with col2:
                    custom_distribution['C+'] = st.slider('C+ Grade %', 0, 100, 15)
                    custom_distribution['C'] = st.slider('C Grade %', 0, 100, 10)
                    custom_distribution['C-'] = st.slider('C- Grade %', 0, 100, 10)
                    custom_distribution['D'] = st.slider('D Grade %', 0, 100, 5)
                
                total = sum(custom_distribution.values())
                st.write(f"Total percentage: {total}%")
                
                if total != 100:
                    st.error("Percentages must sum to 100!")
                    st.stop()
                
                # Convert to proportions
                custom_distribution = {k: v/100 for k, v in custom_distribution.items()}
            
            if st.button("Process Grades"):
                results = pd.DataFrame()
                results[['first_name', 'last_name', 'email']] = data[['first_name', 'last_name', 'email']]
                
                # Process grades for each subject
                for subject in st.session_state.grading_system.subject_columns:
                    scores = data[subject].values
                    grades = st.session_state.grading_system.process_grades(
                        scores,
                        'HEC' if grading_method == "HEC Relative Grading" else 'CUSTOM',
                        custom_distribution
                    )
                    results[f'{subject}_grade'] = grades
                
                # Calculate CGPA
                st.subheader("Results")
                
                # Display grade distribution
                st.write("Grade Distribution by Subject")
                fig, ax = plt.subplots(figsize=(12, 6))
                grade_counts = pd.DataFrame()
                
                for subject in st.session_state.grading_system.subject_columns:
                    grade_counts[subject.replace('_score', '')] = results[f'{subject}_grade'].value_counts()
                
                grade_counts.plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display detailed results
                st.write("Detailed Results")
                st.dataframe(results)
                
                # Save results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="grading_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
