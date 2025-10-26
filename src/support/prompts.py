TEST_DATA_PROMPT = """
    ### TEST DATA GENERATION
    you are a 6th grader student with an assignment to choose a topic and build a 1000 word essay of that subject. I want you to build 3 different essays with following: Essay 1 - should be concise and well written, Essay 2 - should contain some grammar errors but should be a passing grade essay, Essay 3 should be plagiarized and be less relevant to the source. Make sure to include different names for each essay. The output should look as follows:
    Name: John Doe
    Date: 2025-08-25
    Class: 10
    Subject: English
    Source: URL_TO_SOURCE

    [ESSAY CONTENT GOES HERE]
###
"""

PLAGIARISM_CHECK = """
    Analyze the following assignment for potential plagiarism.
    Identify any phrases or sections that appear copied or unoriginal.
    Provide a brief summary of your findings and a plagiarism likelihood score (0–100).

    Assignment:
    {text}
    """

GRAMMAR_CHECK = """Count grammatical errors in the following:
{text}"""

RELEVANCE_CHECK = """Analyze how well the assignment relates to the provided source material. Focus only on evaluation, not suggestions for improvement.

Evaluate the following aspects:
1. Does the assignment address the main topics from the source?
2. Are the facts presented consistent with the source material?
3. How closely does the assignment content align with the source?

Provide only an analytical assessment without offering revisions or examples of improvements.

Assignment:
{text}

Source:
{source}"""

GRADING_PROMPT = """You are an academic evaluator. Grade the following student assignment based on four criteria:
1. Factual Accuracy (0–10): How accurate is the content compared to the source?
2. Relevance to Source (0–10): How well does the assignment relate to the source material?
3. Coherence (0–10): How well-structured and logical is the writing?
4. Grammar (1–10): How well is the assignment written? Evaluate spelling, grammar, sentence structure, and writing quality. Score should reflect writing proficiency regardless of content accuracy. Minimum score is 1.

Assignment:
{answer}

Source Material:
{source}

Return your response as a JSON object like:
{{
  "factuality": float,
  "relevance": float,
  "coherence": float,
  "grammar": float
}}
Only return the JSON. Do not include any explanation or formatting."""

SUMMARY_PROMPT = """Summarize this assignment in 2–3 sentences:

{text}"""
