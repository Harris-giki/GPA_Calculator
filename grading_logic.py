import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from scipy import stats
import logging
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from typing import Dict, List, Tuple
import sys
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

    def load_data(self, file):
        """Load data from CSV file."""
        try:
            if isinstance(file, str):
                self.data = pd.read_csv(file)
            else:
                self.data = pd.read_csv(file)
            
            # Get subject columns
            self.subject_columns = [col for col in self.data.columns if col.endswith('_score')]
            
            if not self.subject_columns:
                raise ValueError("No score columns found in the data")
                
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def set_credit_hours(self) -> None:
        """Get credit hours for each subject from user input."""
        print("\nEnter credit hours for each subject:")
        for subject in self.subject_columns:
            while True:
                try:
                    hours = float(input(f"{subject.replace('_score', '')}: "))
                    if hours <= 0:
                        raise ValueError("Credit hours must be positive")
                    self.credit_hours[subject] = hours
                    break
                except ValueError as e:
                    print(f"Invalid input: {str(e)}. Please try again.")

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
        # Convert to numpy array and ensure float type
        scores = np.array(scores, dtype=float)
        
        # Calculate mean and standard deviation
        mean = np.mean(scores)
        std = np.std(scores)
        
        # Initialize grades list
        grades = []
        
        # Calculate z-scores and assign grades
        for score in scores:
            z_score = (score - mean) / std
            
            # Assign grades based on z-score
            if z_score > 2.0:
                grades.append('A')
            elif z_score > 1.5:
                grades.append('B+')
            elif z_score > 1.0:
                grades.append('B')
            elif z_score > 0.5:
                grades.append('B-')
            elif z_score > 0.0:
                grades.append('C+')
            elif z_score > -0.5:
                grades.append('C')
            elif z_score > -1.0:
                grades.append('C-')
            else:
                grades.append('D')
        
        return grades

    def apply_custom_relative_grading(self, scores: np.array, std_thresholds: Dict[str, Tuple[float, float]]) -> List[str]:
        """
        Apply custom relative grading based on user-defined standard deviation thresholds.
    
        Args:
        scores (np.array): Array of student scores
        std_thresholds (Dict[str, Tuple[float, float]]): Dictionary mapping grades to (lower, upper) 
            standard deviation bounds from mean
    
    Returns:
        List[str]: List of grades
    """
    # Calculate mean and standard deviation
        mean = np.mean(scores)
        std = np.std(scores)
    
    # Initialize grades list
        grades = []
    
    # Convert scores to z-scores and assign grades
        for score in scores:
            z_score = (score - mean) / std
        
        # Find appropriate grade based on z-score
            grade_assigned = False
            for grade, (lower_bound, upper_bound) in std_thresholds.items():
                if lower_bound <= z_score <= upper_bound:
                    grades.append(grade)
                    grade_assigned = True
                    break
        
        # Assign D grade if no threshold matched
            if not grade_assigned:
                grades.append('D')
    
    # Log grade distribution
        grade_counts = pd.Series(grades).value_counts()
        total_students = len(grades)
        logging.info("\nGrade Distribution:")
        for grade, count in grade_counts.items():
            percentage = (count/total_students) * 100
            logging.info(f"{grade}: {count} students ({percentage:.2f}%)")
    
        return grades

    def get_std_thresholds() -> Dict[str, Tuple[float, float]]:
        """Get standard deviation thresholds for each grade from user input."""
        std_thresholds = {}
        grades = ['A*', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-']
    
        print("\nEnter standard deviation thresholds for each grade.")
        print("Example: For grade A, if you enter '1.5 2.0', it means:")
        print("Scores between 1.5 and 2.0 standard deviations above mean will get an A")
        print("Note: Ensure thresholds are continuous and non-overlapping")
    
        previous_lower = float('inf')
        for grade in grades:
            while True:
                try:
                    bounds = input(f"\nEnter lower and upper bounds for {grade} (separated by space): ").split()
                    if len(bounds) != 2:
                        print("Please enter two numbers separated by space")
                        continue
                    
                    upper, lower = float(bounds[0]), float(bounds[1])
                
                    # Validate bounds
                    if upper <= lower:
                        print("Upper bound must be greater than lower bound")
                        continue
                    
                    if lower >= previous_lower:
                        print(f"Lower bound must be less than previous grade's lower bound ({previous_lower})")
                        continue
                
                    std_thresholds[grade] = (lower, upper)
                    previous_lower = lower
                    break
                
                except ValueError:
                    print("Please enter valid numbers")
    
    # Add D grade threshold automatically
        lowest_bound = min(lower for lower, _ in std_thresholds.values())
        std_thresholds['D'] = (float('-inf'), lowest_bound)
    
        return std_thresholds

    def process_grades(self, method="HEC", custom_thresholds=None):
        """Process grades using specified method."""
        try:
            # Create a copy of the dataframe
            results = self.data.copy()
            
            # Process each subject
            for col in self.subject_columns:
                # Get scores as numpy array
                scores = np.array(results[col].values, dtype=float)
                
                # Apply grading method
                if method == "CUSTOM":
                    if not custom_thresholds:
                        raise ValueError("Custom thresholds required for custom grading")
                    grades = self.apply_custom_relative_grading(scores, custom_thresholds)
                else:
                    grades = self.apply_hec_relative_grading(scores)
                
                # Add grades to results
                grade_col = col.replace('_score', '_grade')
                results[grade_col] = grades
            
            # Calculate GPA
            results['gpa'] = self.calculate_gpa(results)
            
            return results
        
        except Exception as e:
            raise Exception(f"Error calculating grades: {str(e)}")

    def calculate_gpa(self, results: pd.DataFrame) -> pd.Series:
        """Calculate GPA for each student."""
        # Define grade points
        grade_points = {
            'A': 4.00, 'A-': 3.67, 'B+': 3.33, 'B': 3.00,
            'B-': 2.67, 'C+': 2.33, 'C': 2.00, 'C-': 1.67,
            'D': 1.00, 'F': 0.00
        }
        
        # Initialize series for total points and credits
        total_points = pd.Series(0.0, index=results.index)
        total_credits = 0
        
        # Calculate for each subject
        for subject in self.subject_columns:
            grade_col = subject.replace('_score', '_grade')
            credits = self.credit_hours.get(subject, 3)  # default to 3 if not specified
            total_credits += credits
            
            # Convert grades to points and multiply by credits
            subject_points = results[grade_col].map(lambda x: grade_points.get(x, 0.0)) * credits
            total_points += subject_points
        
        # Calculate GPA
        if total_credits > 0:
            gpa = (total_points / total_credits).round(2)
        else:
            gpa = pd.Series(0.0, index=results.index)
        
        return gpa

    def visualize_distribution(self, original_grades: List[str], adjusted_grades: List[str],
                             subject: str, method: str) -> None:
        """Create visualization comparing original and adjusted grade distributions."""
        plt.figure(figsize=(15, 8))

        # Plot 1: Grade Distribution
        plt.subplot(1, 3, 1)
        grade_order = ['A*', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D']

        adj_grade_counts = pd.Series(adjusted_grades).value_counts().reindex(grade_order).fillna(0)
        plt.bar(range(len(grade_order)), adj_grade_counts.values, alpha=0.7)
        plt.xticks(range(len(grade_order)), grade_order, rotation=45)
        plt.title(f'{subject} Grade Distribution\n({method})')
        plt.xlabel('Grades')
        plt.ylabel('Number of Students')

        # Plot 2: Score Distribution with Grade Boundaries
        plt.subplot(1, 3, 2)
        scores = self.data[f'{subject}_score'].values
        mean = np.mean(scores)
        std = np.std(scores)

        # Plot score distribution
        sns.histplot(scores, kde=True)

        # Add vertical lines for grade boundaries
        boundaries = [
            (mean + 2*std, 'A*'),
            (mean + (3/2)*std, 'A'),
            (mean + std, 'A-'),
            (mean + std/2, 'B+'),
            (mean, 'B'),
            (mean - std/2, 'B-'),
            (mean - std, 'C+'),
            (mean - (4/3)*std, 'C'),
            (mean - (5/3)*std, 'C-'),
            (mean - 2*std, 'D')
        ]

        for boundary, grade in boundaries:
            plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.3)
            plt.text(boundary, plt.ylim()[1], grade, rotation=90, verticalalignment='top')

        plt.title('Score Distribution with Grade Boundaries')
        plt.xlabel('Scores')
        plt.ylabel('Frequency')

        # Plot 3: Normal Probability Plot
        plt.subplot(1, 3, 3)
        stats.probplot(scores, dist="norm", plot=plt)
        plt.title('Normal Probability Plot')

        plt.tight_layout()
        plt.savefig(f'{subject}_grade_distribution.png')
        plt.close()

    def visualize_results(self, hec_results: pd.DataFrame = None,
                         custom_results: pd.DataFrame = None,
                         save_path: str = 'visualization_results') -> None:
        """Create visualizations for grade distribution and GPA comparison"""
        # Create main directory and subdirectories
        os.makedirs(save_path, exist_ok=True)
        if hec_results is not None:
            os.makedirs(os.path.join(save_path, 'hec'), exist_ok=True)
        if custom_results is not None:
            os.makedirs(os.path.join(save_path, 'custom'), exist_ok=True)


        try:
            plt.style.use('ggplot')

            if custom_results is not None and hec_results is not None:
            # GPA Distribution Comparison
              plt.figure(figsize=(15, 6))
              plt.subplot(1, 2, 1)
              sns.histplot(data=hec_results, x='gpa', bins=20, label='HEC', alpha=0.5)
              sns.histplot(data=custom_results, x='gpa', bins=20, label='Custom', alpha=0.5)
              plt.title('GPA Distribution Comparison')
              plt.legend()

            # Grade Distribution Comparison
              plt.subplot(1, 2, 2)
              grade_cols = [col for col in hec_results.columns if col.endswith('_grade')]
              hec_grades = pd.concat([hec_results[col].value_counts() for col in grade_cols], axis=1)
              custom_grades = pd.concat([custom_results[col].value_counts() for col in grade_cols], axis=1)

              comparison_df = pd.DataFrame({
                  'HEC': hec_grades.mean(axis=1),
                  'Custom': custom_grades.mean(axis=1)
              })
              comparison_df.plot(kind='bar')
              plt.title('Average Grade Distribution')
              plt.tight_layout()
              plt.savefig(os.path.join(save_path, 'gpa_and_grade_comparison.png'))
              plt.close()

            # Generate subject analysis for both
              print("\nHEC Grading Analysis:")
              self.visualize_subject_analysis(hec_results, os.path.join(save_path, 'hec'))

              print("\nCustom Grading Analysis:")
              self.visualize_subject_analysis(custom_results, os.path.join(save_path, 'custom'))

            else:
                results_df = hec_results if hec_results is not None else custom_results

            # GPA Distribution
                plt.figure(figsize=(15, 6))
                plt.subplot(1, 2, 1)
                sns.histplot(data=results_df, x='gpa', bins=20)
                plt.title('GPA Distribution')

            # Overall Grade Distribution
                plt.subplot(1, 2, 2)
                grade_cols = [col for col in results_df.columns if col.endswith('_grade')]
                grade_counts = pd.concat([results_df[col].value_counts() for col in grade_cols], axis=1)
                grade_counts.mean(axis=1).sort_index().plot(kind='bar')
                plt.title('Average Grade Distribution')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'gpa_and_grade_distribution.png'))
                plt.close()

            # Generate subject analysis
                self.visualize_subject_analysis(results_df, save_path)

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")


class GPACalculator:
    def __init__(self):
        # HEC predefined grade thresholds
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
        self.subject_credits = {}
        self.df = None

    def load_data(self, file_path: str) -> None:
        """Load student data from CSV file"""
        self.df = pd.read_csv(file_path)
        print(f"Loaded data for {len(self.df)} students")

    def get_subject_credits(self) -> None:
        """Get credit hours for each subject from user input"""
        subjects = ['math', 'history', 'physics', 'chemistry', 'biology', 'english', 'geography']
        for subject in subjects:
            while True:
                try:
                    credits = int(input(f"Enter credit hours for {subject}: "))
                    if credits > 0:
                        self.subject_credits[subject] = credits
                        break
                    else:
                        print("Credit hours must be positive")
                except ValueError:
                    print("Please enter a valid number")

    def get_custom_thresholds(self) -> Dict[str, Tuple[int, int, float]]:
      """Get custom grade thresholds from user input with robust validation"""
      custom_thresholds = {}
      grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D']
      used_points = set()
      previous_min = 100  # Start from 100 as the highest possible score

      print("\nEnter custom grade thresholds:")
      print("For each grade, enter minimum score, maximum score, and grade points")
      print("Note: Intervals must not overlap and grade points must be unique")
      print("Score range: 0-100, Grade points range: 0-4.0")

      for grade in grades:
         while True:
              try:
                  print(f"\nSetting thresholds for grade {grade}")
                  print(f"Previous minimum score: {previous_min}")
                  if used_points:
                      print("Already used grade points:", sorted(used_points, reverse=True))

                # Get maximum score
                  while True:
                      max_score = float(input(f"{grade} - Enter maximum score (must be less than {previous_min}): "))
                      if 0 <= max_score < previous_min:
                          break
                      print(f"Error: Maximum score must be between 0 and {previous_min}")

                # Get minimum score
                  while True:
                      min_score = float(input(f"{grade} - Enter minimum score (must be less than {max_score}): "))
                      if 0 <= min_score < max_score:
                          break
                      print(f"Error: Minimum score must be between 0 and {max_score}")

                  # Get grade points
                  while True:
                      points = float(input(f"{grade} - Enter grade points (0-4.0, not used before): "))
                      if 0 <= points <= 4.0 and points not in used_points:
                          break
                      elif points in used_points:
                          print(f"Error: Grade point {points} has already been used")
                      else:
                          print("Error: Grade points must be between 0 and 4.0")

                  # Update tracking variables
                  custom_thresholds[grade] = (min_score, max_score, points)
                  used_points.add(points)
                  previous_min = min_score  # Update for next iteration
                  break

              except ValueError:
                  print("Please enter valid numbers")

    # Add F grade automatically based on the lowest threshold
      lowest_min = min(min_score for min_score, _, _ in custom_thresholds.values())
      custom_thresholds['F'] = (0, lowest_min - 0.01, 0.00)

      # Display final thresholds for verification
      print("\nFinal Grade Thresholds:")
      print("-" * 50)
      print("Grade | Score Range | Grade Points")
      print("-" * 50)
      for grade in ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F']:
          min_score, max_score, points = custom_thresholds[grade]
          print(f"{grade:^5} | {min_score:>5.1f}-{max_score:<5.1f} | {points:.2f}")
      print("-" * 50)

      # Ask for confirmation
      while True:
          confirm = input("\nAre these thresholds correct? (yes/no): ").lower()
          if confirm == 'yes':
              return custom_thresholds
          elif confirm == 'no':
              print("\nLet's set the thresholds again.")
              return self.get_custom_thresholds()
          else:
              print("Please enter 'yes' or 'no'")

    def validate_thresholds(self, thresholds: Dict[str, Tuple[int, int, float]]) -> bool:
      """Validate that thresholds don't overlap and grade points are unique"""
    # Sort thresholds by max score
      sorted_grades = sorted(thresholds.items(),
                         key=lambda x: x[1][1],  # Sort by max score
                         reverse=True)

    # Check for overlaps and unique points
      points_set = set()
      previous_min = None

      for grade, (min_score, max_score, points) in sorted_grades:
          # Check score range
        if not (0 <= min_score <= max_score <= 100):
            print(f"Invalid score range for grade {grade}: {min_score}-{max_score}")
            return False

        # Check for overlapping ranges
        if previous_min is not None and min_score >= previous_min:
            print(f"Overlapping range detected at grade {grade}")
            return False

        # Check for duplicate points
        if points in points_set:
            print(f"Duplicate grade points detected: {points}")
            return False

        points_set.add(points)
        previous_min = min_score

      return True

    def calculate_grade(self, score: float, thresholds: Dict[str, Tuple[int, int, float]]) -> str:
        """Calculate letter grade based on score"""
        for grade, (min_score, max_score, _) in thresholds.items():
            if min_score <= score <= max_score:
                return grade
        return 'F'

    def calculate_grade_points(self, grade: str, thresholds: Dict[str, Tuple[int, int, float]]) -> float:
        """Convert letter grade to grade points"""
        for grade_key, (_, _, points) in thresholds.items():
            if grade == grade_key:
                return points
        return 0.0

    def calculate_results(self, thresholds: Dict[str, Tuple[int, int, float]]) -> pd.DataFrame:
        """Calculate grades and GPA for all students"""
        results_df = self.df.copy()
        subject_columns = ['math_score', 'history_score', 'physics_score',
                         'chemistry_score', 'biology_score', 'english_score',
                         'geography_score']

        # Calculate grades for each subject
        for subject in subject_columns:
            subject_name = subject.split('_')[0]
            grade_column = f"{subject_name}_grade"
            results_df[grade_column] = results_df[subject].apply(
                lambda x: self.calculate_grade(x, thresholds))

        # Calculate GPA
        total_credits = sum(self.subject_credits.values())
        weighted_grades = 0

        for subject in subject_columns:
            subject_name = subject.split('_')[0]
            grade_column = f"{subject_name}_grade"
            credit_hours = self.subject_credits[subject_name]

            weighted_grades += results_df[grade_column].apply(
                lambda x: self.calculate_grade_points(x, thresholds)) * credit_hours

        results_df['gpa'] = weighted_grades / total_credits

        # Display student names and GPAs
        print("\nStudent GPAs:")
        print("-" * 50)
        student_results = results_df[['first_name', 'last_name', 'gpa']].sort_values('gpa', ascending=False)
        for _, row in student_results.iterrows():
            print(f"{row['first_name']} {row['last_name']}: {row['gpa']:.2f}")
        print("-" * 50)

        return results_df

    def generate_statistics(self, results_df: pd.DataFrame) -> Dict:
        """Generate statistical analysis of grades and scores"""
        stats = {}

        # Score statistics for each subject
        score_columns = [col for col in results_df.columns if col.endswith('_score')]
        for col in score_columns:
            stats[f"{col}_stats"] = {
                'mean': results_df[col].mean(),
                'median': results_df[col].median(),
                'std': results_df[col].std(),
                'min': results_df[col].min(),
                'max': results_df[col].max()
            }

        # Grade distribution
        grade_columns = [col for col in results_df.columns if col.endswith('_grade')]
        for col in grade_columns:
            stats[f"{col}_distribution"] = results_df[col].value_counts().sort_index().to_dict()

        # GPA statistics
        stats['gpa_stats'] = {
            'mean': results_df['gpa'].mean(),
            'median': results_df['gpa'].median(),
            'std': results_df['gpa'].std(),
            'min': results_df['gpa'].min(),
            'max': results_df['gpa'].max()
        }

        return stats

    def visualize_subject_analysis(self, results_df: pd.DataFrame, save_path: str) -> None:
        """Create and save detailed subject-wise visualizations"""
        # Create all necessary directories
        os.makedirs(save_path, exist_ok=True)

        plt.style.use('ggplot')
        subjects = ['math', 'history', 'physics', 'chemistry', 'biology', 'english', 'geography']

        try:
            # 1. Score Distribution for All Subjects (Boxplot)
            plt.figure(figsize=(15, 8))
            score_data = [results_df[f'{subject}_score'] for subject in subjects]
            plt.boxplot(score_data, labels=subjects)
            plt.title('Score Distribution Across Subjects')
            plt.ylabel('Scores')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'score_distribution_boxplot.png'))
            plt.close()

            # 2. Grade Distribution for Each Subject
            plt.figure(figsize=(15, 8))
            grade_counts = pd.DataFrame()
            for subject in subjects:
                grade_counts[subject] = results_df[f'{subject}_grade'].value_counts()

            grade_counts.plot(kind='bar')
            plt.title('Grade Distribution by Subject')
            plt.xlabel('Grades')
            plt.ylabel('Number of Students')
            plt.legend(title='Subjects', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'grade_distribution_by_subject.png'))
            plt.close()

            # 3. Score Histograms for Each Subject
            fig, axes = plt.subplots(3, 3, figsize=(20, 20))
            axes = axes.ravel()

            for idx, subject in enumerate(subjects):
                sns.histplot(data=results_df, x=f'{subject}_score', bins=20, ax=axes[idx])
                axes[idx].set_title(f'{subject.capitalize()} Score Distribution')
                axes[idx].set_xlabel('Score')
                axes[idx].set_ylabel('Count')

            if len(axes) > len(subjects):
                for idx in range(len(subjects), len(axes)):
                    fig.delaxes(axes[idx])

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'subject_score_distributions.png'))
            plt.close()

            # 4. Correlation Heatmap between Subjects
            plt.figure(figsize=(12, 10))
            score_columns = [col for col in results_df.columns if col.endswith('_score')]
            correlation_matrix = results_df[score_columns].corr()

            sns.heatmap(correlation_matrix,
                       annot=True,
                       cmap='coolwarm',
                       center=0,
                       fmt='.2f',
                       xticklabels=[col.split('_')[0] for col in score_columns],
                       yticklabels=[col.split('_')[0] for col in score_columns])
            plt.title('Subject Score Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'subject_correlations.png'))
            plt.close()

        except Exception as e:
            print(f"Error saving visualizations: {str(e)}")
            return

        # Print analysis in terminal
        print("\nSubject-wise Analysis:")
        print("-" * 50)
        for subject in subjects:
            scores = results_df[f'{subject}_score']
            grades = results_df[f'{subject}_grade']

            print(f"\n{subject.upper()} Statistics:")
            print(f"Average Score: {scores.mean():.2f}")
            print(f"Median Score: {scores.median():.2f}")
            print(f"Standard Deviation: {scores.std():.2f}")
            print(f"Grade Distribution:")
            grade_dist = grades.value_counts().sort_index()
            for grade, count in grade_dist.items():
                print(f"  {grade}: {count} students ({(count/len(grades)*100):.1f}%)")

    def visualize_results(self, hec_results: pd.DataFrame = None,
                         custom_results: pd.DataFrame = None,
                         save_path: str = 'visualization_results') -> None:
        """Create visualizations for grade distribution and GPA comparison"""
        # Create main directory and subdirectories
        os.makedirs(save_path, exist_ok=True)
        if hec_results is not None:
            os.makedirs(os.path.join(save_path, 'hec'), exist_ok=True)
        if custom_results is not None:
            os.makedirs(os.path.join(save_path, 'custom'), exist_ok=True)


        try:
            plt.style.use('ggplot')

            if custom_results is not None and hec_results is not None:
            # GPA Distribution Comparison
              plt.figure(figsize=(15, 6))
              plt.subplot(1, 2, 1)
              sns.histplot(data=hec_results, x='gpa', bins=20, label='HEC', alpha=0.5)
              sns.histplot(data=custom_results, x='gpa', bins=20, label='Custom', alpha=0.5)
              plt.title('GPA Distribution Comparison')
              plt.legend()

            # Grade Distribution Comparison
              plt.subplot(1, 2, 2)
              grade_cols = [col for col in hec_results.columns if col.endswith('_grade')]
              hec_grades = pd.concat([hec_results[col].value_counts() for col in grade_cols], axis=1)
              custom_grades = pd.concat([custom_results[col].value_counts() for col in grade_cols], axis=1)

              comparison_df = pd.DataFrame({
                  'HEC': hec_grades.mean(axis=1),
                  'Custom': custom_grades.mean(axis=1)
              })
              comparison_df.plot(kind='bar')
              plt.title('Average Grade Distribution')
              plt.tight_layout()
              plt.savefig(os.path.join(save_path, 'gpa_and_grade_comparison.png'))
              plt.close()

            # Generate subject analysis for both
              print("\nHEC Grading Analysis:")
              self.visualize_subject_analysis(hec_results, os.path.join(save_path, 'hec'))

              print("\nCustom Grading Analysis:")
              self.visualize_subject_analysis(custom_results, os.path.join(save_path, 'custom'))

            else:
                results_df = hec_results if hec_results is not None else custom_results

            # GPA Distribution
                plt.figure(figsize=(15, 6))
                plt.subplot(1, 2, 1)
                sns.histplot(data=results_df, x='gpa', bins=20)
                plt.title('GPA Distribution')

            # Overall Grade Distribution
                plt.subplot(1, 2, 2)
                grade_cols = [col for col in results_df.columns if col.endswith('_grade')]
                grade_counts = pd.concat([results_df[col].value_counts() for col in grade_cols], axis=1)
                grade_counts.mean(axis=1).sort_index().plot(kind='bar')
                plt.title('Average Grade Distribution')
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'gpa_and_grade_distribution.png'))
                plt.close()

            # Generate subject analysis
                self.visualize_subject_analysis(results_df, save_path)

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
''''
def main():
    print("\nWELCOME TO THE GRADING SYSTEM")
    print("=" * 40)

    # Choose grading system type
    while True:
        system_choice = input("\nChoose grading system type:\n1. Relative Grading\n2. Absolute Grading\nEnter your choice (1/2): ")
        if system_choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")

    if system_choice == '1':
        # Relative Grading System
        grading_system = GradingSystem()

        print("\nGPA CALCULATION AND RELATIVE GRADING SYSTEM")
        print("=" * 40)

        # Load data
        try:
            file_path = input("\nEnter the path to your CSV file: ")
            grading_system.load_data(file_path)
            print(f"Successfully loaded data for {len(grading_system.data)} students")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            sys.exit(1)

        # Get credit hours
        print("\nEnter credit hours for each subject:")
        print("-" * 40)
        grading_system.set_credit_hours()

        # Choose grading method
        while True:
            method = input("\nChoose grading method (HEC/custom): ").upper()
            if method in ['HEC', 'CUSTOM']:
                break
            print("Invalid choice. Please enter either 'HEC' or 'custom'")

        custom_distribution = None
        if method == 'CUSTOM':
            print("\nEnter percentage distribution for grades (should sum to 100):")
            print("-" * 40)
            custom_distribution = {}
            total = 0
            for grade in ['A', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D']:
                while True:
                    try:
                        percentage = float(input(f"Percentage for {grade}: "))
                        if percentage < 0:
                            raise ValueError("Percentage cannot be negative")
                        total += percentage
                        if total > 100:
                            raise ValueError("Total percentage exceeds 100")
                        custom_distribution[grade] = percentage / 100
                        break
                    except ValueError as e:
                        print(f"Invalid input: {str(e)}. Please try again.")

        # Process grades
        try:
            results = grading_system.process_grades(method, custom_distribution)
        except Exception as e:
            logging.error(f"Error processing grades: {str(e)}")
            print(f"An error occurred: {str(e)}")

    else:
        # Absolute Grading System
        calculator = GPACalculator()

        # Step 1: Load data
        file_path = input("Enter the path to your CSV file: ")
        calculator.load_data(file_path)

        # Step 2: Get credit hours
        calculator.get_subject_credits()

        # Step 3: Choose grading method
        while True:
            method = input("\nChoose grading method (1 for HEC, 2 for Custom): ")
            if method in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")

        # Create visualization directory
        os.makedirs('visualization_results', exist_ok=True)

        # Step 4: Calculate and display results
        if method == '1':
            # Use HEC thresholds
            results = calculator.calculate_results(calculator.hec_thresholds)
            stats = calculator.generate_statistics(results)

            print("\nResults using HEC thresholds:")
            print("\nGPA Statistics:")
            for key, value in stats['gpa_stats'].items():
                print(f"{key}: {value:.2f}")

            # Show and save visualizations
            calculator.visualize_results(hec_results=results, save_path='visualization_results')

            # Save results to CSV
            results.to_csv('hec_grading_results.csv', index=False)
            print("\nResults saved to 'hec_grading_results.csv'")
            print("Visualizations saved in 'visualization_results' directory")

            # Display subject-wise performance summary
            print_subject_performance(results)

        else:
            # Get custom thresholds and calculate both
            custom_thresholds = calculator.get_custom_thresholds()
            hec_results = calculator.calculate_results(calculator.hec_thresholds)
            custom_results = calculator.calculate_results(custom_thresholds)

            hec_stats = calculator.generate_statistics(hec_results)
            custom_stats = calculator.generate_statistics(custom_results)

            print("\nComparison of Results:")
            print("\nHEC Grading Statistics:")
            for key, value in hec_stats['gpa_stats'].items():
                print(f"{key}: {value:.2f}")

            print("\nCustom Grading Statistics:")
            for key, value in custom_stats['gpa_stats'].items():
                print(f"{key}: {value:.2f}")

            # Show and save visualizations for both methods
            calculator.visualize_results(hec_results=hec_results,
                                      custom_results=custom_results,
                                      save_path='visualization_results')

            # Save results to CSV
            hec_results.to_csv('hec_grading_results.csv', index=False)
            custom_results.to_csv('custom_grading_results.csv', index=False)
            print("\nResults saved to 'hec_grading_results.csv' and 'custom_grading_results.csv'")
            print("Visualizations saved in 'visualization_results' directory")

            # Display subject-wise performance summary for both methods
            for method_name, results in [("HEC", hec_results), ("Custom", custom_results)]:
                print(f"\n{method_name} Grading - Subject Performance Summary:")
                print_subject_performance(results)

def print_subject_performance(results):
    """Helper function to print subject-wise performance summary"""
    subject_cols = ['math_score', 'history_score', 'physics_score',
                   'chemistry_score', 'biology_score', 'english_score',
                   'geography_score']

    for col in subject_cols:
        subject = col.split('_')[0]
        print(f"\n{subject.upper()}:")
        print(f"Average Score: {results[col].mean():.2f}")
        print(f"Grade Distribution:")
        grade_dist = results[f"{subject}_grade"].value_counts().sort_index()
        for grade, count in grade_dist.items():
            print(f"  {grade}: {count} students ({(count/len(results)*100):.1f}%)")

if __name__ == "__main__":
    main()

'''