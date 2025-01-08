import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from train_model import FaceDetector, train_model

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
        try:
            self.data = pd.read_csv(file)
            self.subject_columns = [col for col in self.data.columns if col.endswith('_score')]
            if not self.subject_columns:
                raise ValueError("No score columns found in the data")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def apply_hec_relative_grading(self, scores):
        scores = np.array(scores, dtype=float)
        mean = np.mean(scores)
        std = np.std(scores)
        grades = []
        
        for score in scores:
            z_score = (score - mean) / std
            if z_score > 2.0: grades.append('A')
            elif z_score > 1.5: grades.append('B+')
            elif z_score > 1.0: grades.append('B')
            elif z_score > 0.5: grades.append('B-')
            elif z_score > 0.0: grades.append('C+')
            elif z_score > -0.5: grades.append('C')
            elif z_score > -1.0: grades.append('C-')
            else: grades.append('D')
        
        return grades

    def process_grades(self, method="HEC"):
        try:
            results = self.data.copy()
            for col in self.subject_columns:
                scores = np.array(results[col].values, dtype=float)
                grades = self.apply_hec_relative_grading(scores)
                grade_col = col.replace('_score', '_grade')
                results[grade_col] = grades
            
            results['gpa'] = self.calculate_gpa(results)
            return results
        except Exception as e:
            raise Exception(f"Error calculating grades: {str(e)}")

    def calculate_gpa(self, results):
        grade_points = {
            'A': 4.00, 'A-': 3.67, 'B+': 3.33, 'B': 3.00,
            'B-': 2.67, 'C+': 2.33, 'C': 2.00, 'C-': 1.67,
            'D': 1.00, 'F': 0.00
        }
        
        total_points = pd.Series(0.0, index=results.index)
        total_credits = 0
        
        for subject in self.subject_columns:
            grade_col = subject.replace('_score', '_grade')
            credits = self.credit_hours.get(subject, 3)
            total_credits += credits
            subject_points = results[grade_col].map(lambda x: grade_points.get(x, 0.0)) * credits
            total_points += subject_points
        
        return (total_points / total_credits).round(2) if total_credits > 0 else pd.Series(0.0, index=results.index)

class GPACalculator:
    def __init__(self):
        self.hec_thresholds = {
            'A': (85, 100, 4.00), 'A-': (80, 84, 3.66),
            'B+': (75, 79, 3.33), 'B': (71, 74, 3.00),
            'B-': (68, 70, 2.66), 'C+': (64, 67, 2.33),
            'C': (61, 63, 2.00), 'C-': (58, 60, 1.66),
            'D+': (54, 57, 1.30), 'D': (50, 53, 1.00),
            'F': (0, 49, 0.00)
        }
        self.subject_credits = {}
        self.df = None

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)

    def calculate_grade(self, score, thresholds):
        for grade, (min_score, max_score, _) in thresholds.items():
            if min_score <= score <= max_score:
                return grade
        return 'F'

    def calculate_grade_points(self, grade, thresholds):
        for grade_key, (_, _, points) in thresholds.items():
            if grade == grade_key:
                return points
        return 0.0

    def calculate_results(self, thresholds):
        results_df = self.df.copy()
        subject_columns = ['math_score', 'history_score', 'physics_score',
                         'chemistry_score', 'biology_score', 'english_score',
                         'geography_score']

        for subject in subject_columns:
            subject_name = subject.split('_')[0]
            grade_column = f"{subject_name}_grade"
            results_df[grade_column] = results_df[subject].apply(
                lambda x: self.calculate_grade(x, thresholds))

        total_credits = sum(self.subject_credits.values()) if self.subject_credits else len(subject_columns) * 3
        weighted_grades = 0

        for subject in subject_columns:
            subject_name = subject.split('_')[0]
            grade_column = f"{subject_name}_grade"
            credit_hours = self.subject_credits.get(subject_name, 3)
            weighted_grades += results_df[grade_column].apply(
                lambda x: self.calculate_grade_points(x, thresholds)) * credit_hours

        results_df['gpa'] = weighted_grades / total_credits
        return results_df

def predict_student(image_path, model_path='face_recognition_model.pth'):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        num_classes = len(checkpoint['classes'])
        model = FaceDetector(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            student_id = checkpoint['classes'][predicted.item()]
            confidence = probabilities[0][predicted.item()].item()

        return student_id, confidence

    except Exception as e:
        return f"Error during prediction: {str(e)}", 0.0

def process_student_image(image_path, grading_type='absolute'):
    try:
        predicted_name, confidence = predict_student(image_path)
        print(f"DEBUG: Predicted name = {predicted_name}, confidence = {confidence}")
        
        df = pd.read_csv('students.csv')
        print(f"DEBUG: DataFrame columns = {df.columns.tolist()}")
        
        df['full_name'] = df['first_name'] + ' ' + df['last_name']
        student_results = df[df['full_name'] == predicted_name]
        print(f"DEBUG: Found student records: {len(student_results)}")
        print(f"DEBUG: Student data: {student_results.to_dict('records')}")

        if student_results.empty:
            return "Student not found in database"

        if grading_type == 'absolute':
            calculator = GPACalculator()
            calculator.load_data('students.csv')
            calculator.subject_columns = [
                'math_score', 'history_score', 'physics_score',
                'chemistry_score', 'biology_score', 'english_score',
                'geography_score'
            ]
            results = calculator.calculate_results(calculator.hec_thresholds)
        else:
            grading_system = GradingSystem()
            grading_system.load_data('students.csv')
            grading_system.subject_columns = [
                'math_score', 'history_score', 'physics_score',
                'chemistry_score', 'biology_score', 'english_score',
                'geography_score'
            ]
            results = grading_system.process_grades("HEC")

        student_id = student_results['id'].iloc[0]
        
        student_info = {
            'student_id': student_id,
            'name': predicted_name,
            'gpa': results.loc[results['id'] == student_id, 'gpa'].iloc[0],
            'grading_type': grading_type,
            'confidence': confidence
        }
        return student_info

    except Exception as e:
        print(f"Full error details: {str(e)}")
        return f"Error calculating grades: {str(e)}"

if __name__ == "__main__":
    print("Welcome to Face Recognition Based GPA Calculator")
    image_path = input("Enter the path to student image: ")
    
    if not os.path.exists(image_path):
        print("Error: Image file not found")
    else:
        grading_type = input("Choose grading type (absolute/relative): ").lower()
        if grading_type not in ['absolute', 'relative']:
            print("Invalid grading type. Defaulting to absolute grading.")
            grading_type = 'absolute'
            
        result = process_student_image(image_path, grading_type)
        if isinstance(result, dict):
            print("\nResults:")
            print(f"Student ID: {result['student_id']}")
            print(f"Name: {result['name']}")
            print(f"GPA ({result['grading_type']} grading): {result['gpa']:.2f}")
        else:
            print(result)