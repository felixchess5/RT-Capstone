import os

ASSIGNMENTS_FOLDER = 'Assignments'
OUTPUT_FOLDER = './output'
PLAGIARISM_REPORTS_FOLDER = './plagiarism_reports'
GRAPH_OUTPUT_PATH = 'graph.png'
SUMMARY_CSV_PATH = os.path.join(OUTPUT_FOLDER, 'summary.csv')

# Subject-specific output paths
MATH_CSV_PATH = os.path.join(OUTPUT_FOLDER, 'math_assignments.csv')
SPANISH_CSV_PATH = os.path.join(OUTPUT_FOLDER, 'spanish_assignments.csv')
ENGLISH_CSV_PATH = os.path.join(OUTPUT_FOLDER, 'english_assignments.csv')
SCIENCE_CSV_PATH = os.path.join(OUTPUT_FOLDER, 'science_assignments.csv')
HISTORY_CSV_PATH = os.path.join(OUTPUT_FOLDER, 'history_assignments.csv')
GENERAL_CSV_PATH = os.path.join(OUTPUT_FOLDER, 'general_assignments.csv')

# Subject-specific JSON paths
MATH_JSON_PATH = os.path.join(OUTPUT_FOLDER, 'math_assignments.json')
SPANISH_JSON_PATH = os.path.join(OUTPUT_FOLDER, 'spanish_assignments.json')
ENGLISH_JSON_PATH = os.path.join(OUTPUT_FOLDER, 'english_assignments.json')
SCIENCE_JSON_PATH = os.path.join(OUTPUT_FOLDER, 'science_assignments.json')
HISTORY_JSON_PATH = os.path.join(OUTPUT_FOLDER, 'history_assignments.json')
GENERAL_JSON_PATH = os.path.join(OUTPUT_FOLDER, 'general_assignments.json')

REQUIRED_FOLDERS = [
    ASSIGNMENTS_FOLDER,
    OUTPUT_FOLDER,
    PLAGIARISM_REPORTS_FOLDER
]