import unittest
from text_analysis import TextAnalysis

class TestTextAnalysisMethods(unittest.TestCase):

    def setUp(self):
        # Initialize with dummy model paths; replace with actual paths for real testing
        self.analysis = TextAnalysis(
            gpt_model_name='gpt2', 
            word2vec_model_path='path_to_your_word2vec_model.bin', 
            t5_model_path='path_to_your_t5_model',
            device='cpu'
        )

    def test_semantic_complexity_metric(self):
        text = "This is a test sentence."
        complexity = self.analysis.semantic_complexity_metric(text)
        self.assertIsInstance(complexity, float)

    def test_closest_words_word2vec(self):
        word = 'king'
        closest_words = self.analysis.closest_words_word2vec(word, N=5)
        self.assertEqual(len(closest_words), 5)
        self.assertTrue(all(isinstance(word, tuple) for word in closest_words))

    def test_embed(self):
        text = "Test embedding."
        embedding = self.analysis.embed(text)
        self.assertTrue(isinstance(embedding, torch.FloatTensor))

    def test_generate_from_latent(self):
        latent = self.analysis.embed("Test latent generation.")
        sentence = self.analysis.generate_from_latent(latent, max_length=50, temperature=1.0)
        self.assertIsInstance(sentence, str)

    def test_generate_and_analyze_complexity(self):
        anchor_word = 'king'
        N = 3
        sentences_per_pair = 2
        complexity_matrix, generated_sentences = self.analysis.generate_and_analyze_complexity(anchor_word, N, sentences_per_pair)
        self.assertEqual(complexity_matrix.shape, (N, sentences_per_pair))
        self.assertEqual(len(generated_sentences), N)
        self.assertTrue(all(len(sentences) == sentences_per_pair for sentences in generated_sentences.values()))

if __name__ == '__main__':
    unittest.main()
