"""
Tests for GPQAExtractor class.
Tests essential functionality for both raw and processed GPQA data formats.
"""

import pytest
from wisent_guard.core.benchmark_extractors import GPQAExtractor


class TestGPQAExtractor:
    """Essential test cases for GPQAExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = GPQAExtractor()

    def test_raw_format_extraction(self):
        """Test QA pair extraction from raw GPQA format (Idavidrein/gpqa dataset)."""
        raw_doc = {
            'Question': 'Which organelle is primarily responsible for ATP synthesis?',
            'Correct Answer': 'Mitochondria',
            'Incorrect Answer 1': 'Nucleus',
            'Incorrect Answer 2': 'Endoplasmic reticulum',
            'Incorrect Answer 3': 'Golgi apparatus'
        }
        
        result = self.extractor.extract_qa_pair(raw_doc)
        
        assert result is not None
        assert result['question'] == 'Which organelle is primarily responsible for ATP synthesis?'
        assert result['correct_answer'] == 'Mitochondria'
        assert result['answer_letter'] == 'A'
        assert len(result['choices']) == 4
        assert '(A) Mitochondria' in result['formatted_question']

    def test_processed_format_extraction(self):
        """Test QA pair extraction from processed GPQA format (lm-eval transformed)."""
        processed_doc = {
            'Question': 'What is the primary function of chloroplasts?',
            'choice1': 'Protein synthesis',
            'choice2': 'Photosynthesis',  # Correct
            'choice3': 'DNA replication',
            'choice4': 'Cell division',
            'answer': '(B)'
        }
        
        result = self.extractor.extract_qa_pair(processed_doc)
        
        assert result is not None
        assert result['correct_answer'] == 'Photosynthesis'
        assert result['answer_letter'] == 'B'
        assert result['choices'] == ['Protein synthesis', 'Photosynthesis', 'DNA replication', 'Cell division']

    def test_answer_format_variations(self):
        """Test different answer format patterns: (A), (B), (C), (D)."""
        test_cases = [
            ('(A)', 'Answer A', 'A'),
            ('(C)', 'Answer C', 'C'),
            ('(D)', 'Answer D', 'D'),
        ]
        
        for answer_format, expected_answer, expected_letter in test_cases:
            doc = {
                'Question': 'Test question?',
                'choice1': 'Answer A',
                'choice2': 'Answer B', 
                'choice3': 'Answer C',
                'choice4': 'Answer D',
                'answer': answer_format
            }
            
            result = self.extractor.extract_qa_pair(doc)
            assert result['correct_answer'] == expected_answer
            assert result['answer_letter'] == expected_letter

    def test_invalid_data_handling(self):
        """Test graceful handling of invalid or incomplete data."""
        invalid_cases = [
            {},  # Empty document
            {'Question': 'Test?'},  # No answer data
            {'Question': 'Test?', 'choice1': 'A'},  # Incomplete choices
            {'Question': 'Test?', 'choice1': 'A', 'choice2': 'B', 'choice3': 'C', 'choice4': 'D', 'answer': '(Z)'},  # Invalid answer
        ]
        
        for invalid_doc in invalid_cases:
            result = self.extractor.extract_qa_pair(invalid_doc)
            assert result is None

    def test_contrastive_pair_generation(self):
        """Test contrastive pair extraction (correct vs incorrect answers)."""
        doc = {
            'Question': 'Which law of thermodynamics relates to entropy?',
            'choice1': 'First law',
            'choice2': 'Second law',  # Correct
            'choice3': 'Third law', 
            'choice4': 'Zeroth law',
            'answer': '(B)'
        }
        
        result = self.extractor.extract_contrastive_pair(doc)
        
        assert result is not None
        assert result['correct_answer'] == 'Second law'
        assert result['incorrect_answer'] in ['First law', 'Third law', 'Zeroth law']
        assert result['correct_answer'] != result['incorrect_answer']
        assert 'Which law of thermodynamics relates to entropy?' in result['question']

    def test_mixed_format_processing(self):
        """Test processing documents with different formats in batch."""
        extractor = GPQAExtractor()
        
        documents = [
            # Raw format
            {
                'Question': 'Biology question?',
                'Correct Answer': 'Bio Answer',
                'Incorrect Answer 1': 'Wrong Answer 1',
                'Incorrect Answer 2': 'Wrong Answer 2',
                'Incorrect Answer 3': 'Wrong Answer 3'
            },
            # Processed format
            {
                'Question': 'Physics question?',
                'choice1': 'Physics Answer',
                'choice2': 'Wrong Answer',
                'choice3': 'Another Wrong', 
                'choice4': 'Also Wrong',
                'answer': '(A)'
            }
        ]
        
        results = []
        for doc in documents:
            result = extractor.extract_qa_pair(doc)
            if result:
                results.append(result)
        
        assert len(results) == 2
        assert results[0]['correct_answer'] == 'Bio Answer'
        assert results[1]['correct_answer'] == 'Physics Answer'