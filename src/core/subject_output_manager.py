"""
Subject-Specific Output Manager
Handles separate output files for English, Math, Spanish, and other subjects.
"""
import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class OutputSubject(Enum):
    """Supported output subjects."""
    MATHEMATICS = "mathematics"
    SPANISH = "spanish"
    ENGLISH = "english"
    SCIENCE = "science"
    HISTORY = "history"
    GENERAL = "general"

@dataclass
class SubjectOutput:
    """Configuration for subject-specific output."""
    subject: OutputSubject
    csv_filename: str
    json_filename: str
    specialized_fields: List[str]

class SubjectOutputManager:
    """Manages subject-specific output files and formatting."""

    def __init__(self, output_folder: str = "./output"):
        self.output_folder = output_folder
        self.ensure_output_folder()

        # Define subject-specific output configurations
        self.subject_configs = {
            OutputSubject.MATHEMATICS: SubjectOutput(
                subject=OutputSubject.MATHEMATICS,
                csv_filename="math_assignments.csv",
                json_filename="math_assignments.json",
                specialized_fields=[
                    "mathematical_accuracy", "problem_solving_approach",
                    "notation_clarity", "step_by_step_work", "equations_solved",
                    "problem_types", "step_by_step_present", "calculus_operations", "symbolic_solutions"
                ]
            ),
            OutputSubject.SPANISH: SubjectOutput(
                subject=OutputSubject.SPANISH,
                csv_filename="spanish_assignments.csv",
                json_filename="spanish_assignments.json",
                specialized_fields=[
                    "grammar_accuracy", "vocabulary_usage", "fluency_communication",
                    "cultural_understanding", "vocabulary_level", "grammar_errors_count",
                    "cultural_references", "verb_conjugations", "assignment_type_spanish"
                ]
            ),
            OutputSubject.ENGLISH: SubjectOutput(
                subject=OutputSubject.ENGLISH,
                csv_filename="english_assignments.csv",
                json_filename="english_assignments.json",
                specialized_fields=[
                    "literary_analysis", "writing_quality", "thesis_strength",
                    "evidence_support", "style_consistency", "citation_quality"
                ]
            ),
            OutputSubject.SCIENCE: SubjectOutput(
                subject=OutputSubject.SCIENCE,
                csv_filename="science_assignments.csv",
                json_filename="science_assignments.json",
                specialized_fields=[
                    "scientific_accuracy", "hypothesis_quality", "data_analysis",
                    "experimental_design", "conclusion_validity", "subject_area",
                    "assignment_type_science", "scientific_method_completeness",
                    "units_measurements_count", "formulas_identified_count",
                    "data_visualization_present", "scientific_vocabulary_score",
                    "experimental_variables_identified", "safety_considerations_count"
                ]
            ),
            OutputSubject.HISTORY: SubjectOutput(
                subject=OutputSubject.HISTORY,
                csv_filename="history_assignments.csv",
                json_filename="history_assignments.json",
                specialized_fields=[
                    "historical_accuracy", "chronological_understanding", "source_analysis",
                    "contextual_awareness", "argument_development", "historical_period",
                    "assignment_type_history", "region_focus", "dates_identified_count",
                    "historical_figures_count", "events_mentioned_count", "sources_cited_count",
                    "chronological_accuracy", "historical_context_score", "bias_awareness_score",
                    "historical_vocabulary_score", "evidence_usage_score"
                ]
            ),
            OutputSubject.GENERAL: SubjectOutput(
                subject=OutputSubject.GENERAL,
                csv_filename="general_assignments.csv",
                json_filename="general_assignments.json",
                specialized_fields=[
                    "content_quality", "organization", "clarity", "depth_of_analysis"
                ]
            )
        }

    def ensure_output_folder(self):
        """Ensure output folder exists."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def determine_subject(self, assignment_data: Dict) -> OutputSubject:
        """Determine the subject for an assignment based on classification."""
        # Check for specialized processing results
        if "assignment_classification" in assignment_data:
            subject_name = assignment_data["assignment_classification"].get("subject", "general").lower()
        elif "specialized_processing" in assignment_data:
            subject_name = assignment_data["specialized_processing"].get("classification", {}).get("subject", "general").lower()
        elif "processor_used" in assignment_data:
            subject_name = assignment_data["processor_used"].lower()
        else:
            # Fallback to metadata subject
            subject_name = assignment_data.get("subject", "general").lower()

        # Map subject names to OutputSubject enum
        subject_mapping = {
            "mathematics": OutputSubject.MATHEMATICS,
            "math": OutputSubject.MATHEMATICS,
            "algebra": OutputSubject.MATHEMATICS,
            "calculus": OutputSubject.MATHEMATICS,
            "geometry": OutputSubject.MATHEMATICS,
            "spanish": OutputSubject.SPANISH,
            "español": OutputSubject.SPANISH,
            "english": OutputSubject.ENGLISH,
            "literature": OutputSubject.ENGLISH,
            "writing": OutputSubject.ENGLISH,
            "science": OutputSubject.SCIENCE,
            "biology": OutputSubject.SCIENCE,
            "chemistry": OutputSubject.SCIENCE,
            "physics": OutputSubject.SCIENCE,
            "history": OutputSubject.HISTORY,
            "social studies": OutputSubject.HISTORY
        }

        return subject_mapping.get(subject_name, OutputSubject.GENERAL)

    def extract_specialized_data(self, assignment_data: Dict, subject: OutputSubject) -> Dict:
        """Extract subject-specific data from assignment results."""
        specialized_data = {}
        config = self.subject_configs[subject]

        # Extract specialized processing results
        if "specialized_processing" in assignment_data:
            specialized_processing = assignment_data["specialized_processing"]

            if "processing_results" in specialized_processing:
                processing_results = specialized_processing["processing_results"]

                if subject == OutputSubject.MATHEMATICS:
                    # Math-specific data extraction
                    if "grading" in processing_results:
                        grading = processing_results["grading"]
                        specialized_data.update({
                            "mathematical_accuracy": grading.get("mathematical_accuracy", 0),
                            "problem_solving_approach": grading.get("problem_solving_approach", 0),
                            "notation_clarity": grading.get("notation_clarity", 0),
                            "step_by_step_work": grading.get("step_by_step_work", 0)
                        })

                    if "analysis" in processing_results:
                        analysis = processing_results["analysis"]
                        specialized_data.update({
                            "equations_solved": len(analysis.get("equations_found", [])),
                            "problem_types": ", ".join(analysis.get("problem_types", [])),
                            "step_by_step_present": analysis.get("step_by_step_present", False)
                        })

                elif subject == OutputSubject.SPANISH:
                    # Spanish-specific data extraction
                    if "grading" in processing_results:
                        grading = processing_results["grading"]
                        specialized_data.update({
                            "grammar_accuracy": grading.get("grammar_accuracy", 0),
                            "vocabulary_usage": grading.get("vocabulary_usage", 0),
                            "fluency_communication": grading.get("fluency_communication", 0),
                            "cultural_understanding": grading.get("cultural_understanding", 0)
                        })

                    if "analysis" in processing_results:
                        analysis = processing_results["analysis"]
                        specialized_data.update({
                            "vocabulary_level": analysis.get("vocabulary_level", "unknown"),
                            "grammar_errors_count": analysis.get("grammar_errors_count", 0),
                            "cultural_references_count": analysis.get("cultural_references_count", 0),
                            "assignment_type_spanish": analysis.get("assignment_type", "unknown")
                        })

                elif subject == OutputSubject.SCIENCE:
                    # Science-specific data extraction
                    if "grading" in processing_results:
                        grading = processing_results["grading"]
                        specialized_data.update({
                            "scientific_accuracy": grading.get("scientific_accuracy", 0),
                            "hypothesis_quality": grading.get("hypothesis_quality", 0),
                            "data_analysis": grading.get("data_analysis", 0),
                            "experimental_design": grading.get("experimental_design", 0),
                            "conclusion_validity": grading.get("conclusion_validity", 0)
                        })

                    if "analysis" in processing_results:
                        analysis = processing_results["analysis"]
                        specialized_data.update({
                            "subject_area": analysis.get("subject_area", "unknown"),
                            "assignment_type_science": analysis.get("assignment_type", "unknown"),
                            "scientific_method_completeness": analysis.get("scientific_method_elements", {}).get("completeness_percentage", 0),
                            "units_measurements_count": analysis.get("units_and_measurements_count", 0),
                            "formulas_identified_count": analysis.get("formulas_identified_count", 0),
                            "data_visualization_present": analysis.get("data_visualization_present", False),
                            "scientific_vocabulary_score": analysis.get("scientific_vocabulary_score", 0),
                            "experimental_variables_identified": len(analysis.get("experimental_variables", {}).get("independent", [])) + len(analysis.get("experimental_variables", {}).get("dependent", [])),
                            "safety_considerations_count": analysis.get("safety_considerations_count", 0)
                        })

                elif subject == OutputSubject.HISTORY:
                    # History-specific data extraction
                    if "grading" in processing_results:
                        grading = processing_results["grading"]
                        specialized_data.update({
                            "historical_accuracy": grading.get("historical_accuracy", 0),
                            "chronological_understanding": grading.get("chronological_understanding", 0),
                            "source_analysis": grading.get("source_analysis", 0),
                            "contextual_awareness": grading.get("contextual_awareness", 0),
                            "argument_development": grading.get("argument_development", 0)
                        })

                    if "analysis" in processing_results:
                        analysis = processing_results["analysis"]
                        specialized_data.update({
                            "historical_period": analysis.get("period", "unknown"),
                            "assignment_type_history": analysis.get("assignment_type", "unknown"),
                            "region_focus": analysis.get("region_focus", "unknown"),
                            "dates_identified_count": analysis.get("dates_identified_count", 0),
                            "historical_figures_count": analysis.get("historical_figures_count", 0),
                            "events_mentioned_count": analysis.get("events_mentioned_count", 0),
                            "sources_cited_count": analysis.get("sources_cited_count", 0),
                            "chronological_accuracy": analysis.get("chronological_accuracy", 0),
                            "historical_context_score": analysis.get("historical_context_score", 0),
                            "bias_awareness_score": analysis.get("bias_awareness_score", 0),
                            "historical_vocabulary_score": analysis.get("historical_vocabulary_score", 0),
                            "evidence_usage_score": analysis.get("evidence_usage_score", 0)
                        })

        # Add specialized grading if available
        if "specialized_grades" in assignment_data:
            specialized_data.update(assignment_data["specialized_grades"])

        return specialized_data

    def export_subject_csv(self, assignments: List[Dict], subject: OutputSubject) -> str:
        """Export assignments for a specific subject to CSV."""
        config = self.subject_configs[subject]
        csv_path = os.path.join(self.output_folder, config.csv_filename)

        # Filter assignments by subject
        subject_assignments = [
            assignment for assignment in assignments
            if self.determine_subject(assignment) == subject
        ]

        if not subject_assignments:
            print(f"No {subject.value} assignments found to export")
            return ""

        # Define CSV fields
        standard_fields = [
            "student_name", "date_of_submission", "class", "subject",
            "overall_score", "letter_grade", "grammar_errors", "summary"
        ]

        all_fields = standard_fields + config.specialized_fields

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()

            for assignment in subject_assignments:
                # Extract standard data
                row_data = {
                    "student_name": assignment.get("student_name", "Unknown"),
                    "date_of_submission": assignment.get("date_of_submission", "Unknown"),
                    "class": assignment.get("class", "Unknown"),
                    "subject": assignment.get("subject", subject.value),
                    "overall_score": assignment.get("overall_score", "N/A"),
                    "letter_grade": assignment.get("letter_grade", "N/A"),
                    "grammar_errors": assignment.get("grammar_errors", "N/A"),
                    "summary": assignment.get("summary", "No summary available")[:200] + "..." if len(str(assignment.get("summary", ""))) > 200 else assignment.get("summary", "No summary available")
                }

                # Extract specialized data
                specialized_data = self.extract_specialized_data(assignment, subject)
                row_data.update(specialized_data)

                # Fill missing specialized fields with N/A
                for field in config.specialized_fields:
                    if field not in row_data:
                        row_data[field] = "N/A"

                # Remove any fields not in the defined fieldnames to avoid CSV errors
                filtered_row_data = {field: row_data.get(field, "N/A") for field in all_fields}

                writer.writerow(filtered_row_data)

        print(f"✅ Exported {len(subject_assignments)} {subject.value} assignments to {csv_path}")
        return csv_path

    def export_subject_json(self, assignments: List[Dict], subject: OutputSubject) -> str:
        """Export assignments for a specific subject to JSON with full detail."""
        config = self.subject_configs[subject]
        json_path = os.path.join(self.output_folder, config.json_filename)

        # Filter assignments by subject
        subject_assignments = [
            assignment for assignment in assignments
            if self.determine_subject(assignment) == subject
        ]

        if not subject_assignments:
            print(f"No {subject.value} assignments found to export")
            return ""

        # Prepare JSON data with metadata
        json_data = {
            "export_metadata": {
                "subject": subject.value,
                "export_date": datetime.now().isoformat(),
                "total_assignments": len(subject_assignments),
                "average_score": sum(
                    float(assignment.get("overall_score", 0))
                    for assignment in subject_assignments
                    if assignment.get("overall_score") not in ["N/A", None]
                ) / max(len(subject_assignments), 1)
            },
            "assignments": subject_assignments
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Exported {len(subject_assignments)} {subject.value} assignments to {json_path}")
        return json_path

    def export_all_subjects(self, assignments: List[Dict]) -> Dict[str, List[str]]:
        """Export assignments to subject-specific files for all detected subjects."""
        export_results = {}

        # Group assignments by subject
        subject_groups = {}
        for assignment in assignments:
            subject = self.determine_subject(assignment)
            if subject not in subject_groups:
                subject_groups[subject] = []
            subject_groups[subject].append(assignment)

        # Export each subject group
        for subject, subject_assignments in subject_groups.items():
            csv_path = self.export_subject_csv(assignments, subject)
            json_path = self.export_subject_json(assignments, subject)

            export_results[subject.value] = []
            if csv_path:
                export_results[subject.value].append(csv_path)
            if json_path:
                export_results[subject.value].append(json_path)

        # Create summary report
        self.create_export_summary(subject_groups, export_results)

        return export_results

    def create_export_summary(self, subject_groups: Dict, export_results: Dict):
        """Create a summary report of all exported files."""
        summary_path = os.path.join(self.output_folder, "export_summary.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("📊 SUBJECT-SPECIFIC EXPORT SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total_assignments = sum(len(assignments) for assignments in subject_groups.values())
            f.write(f"Total Assignments Processed: {total_assignments}\n\n")

            for subject, assignments in subject_groups.items():
                f.write(f"📚 {subject.value.upper()}\n")
                f.write(f"   Assignments: {len(assignments)}\n")

                if subject.value in export_results:
                    f.write(f"   Files Created:\n")
                    for file_path in export_results[subject.value]:
                        f.write(f"     - {os.path.basename(file_path)}\n")

                # Calculate average score for this subject
                scores = [
                    float(assignment.get("overall_score", 0))
                    for assignment in assignments
                    if assignment.get("overall_score") not in ["N/A", None]
                ]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    f.write(f"   Average Score: {avg_score:.2f}\n")

                f.write("\n")

        print(f"📋 Export summary created: {summary_path}")

def create_subject_output_manager(output_folder: str = "./output") -> SubjectOutputManager:
    """Factory function to create a SubjectOutputManager instance."""
    return SubjectOutputManager(output_folder)