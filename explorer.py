from transformers import GPT2Model, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from gensim.models import KeyedVectors
import torch
import numpy as np

class TextAnalysis:
    def __init__(self, gpt_model_name='gpt2', word2vec_model_path, t5_model_path, device='cpu'):
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.gpt_model = GPT2Model.from_pretrained(gpt_model_name)
        self.word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
        self.device = device
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_path, model_max_length=512)
        self.t5_model = AutoModelForCausalLM.from_pretrained(t5_model_path, trust_remote_code=True).to(self.device)
        self.t5_model.eval()

    def semantic_complexity_metric(self, text):
        inputs = self.gpt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.gpt_model(**inputs)
        return torch.std(outputs.last_hidden_state, dim=1).mean().item()

    def closest_words_word2vec(self, word, N=5):
        return self.word2vec_model.most_similar(positive=[word], topn=N)

    @torch.no_grad()
    def embed(self, text):
        inputs = self.t5_tokenizer(text, return_tensors='pt').to(self.device)
        decoder_inputs = self.t5_tokenizer('', return_tensors='pt').to(self.device)
        return self.t5_model(**inputs, decoder_input_ids=decoder_inputs['input_ids'], encode_only=True)[0]

    @torch.no_grad()
    def generate_from_latent(self, latent, max_length=512, temperature=1.0):
        dummy_text = '.'
        dummy = self.embed(dummy_text)
        perturb_vector = latent - dummy
        self.t5_model.perturb_vector = perturb_vector
        input_ids = self.t5_tokenizer(dummy_text, return_tensors='pt').to(self.device).input_ids
        output = self.t5_model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, temperature=temperature, top_p=0.9, num_return_sequences=1)
        return self.t5_tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_and_analyze_complexity(self, anchor_word, N=5, sentences_per_pair=10):
        closest_words = self.closest_words_word2vec(anchor_word, N)
        complexity_matrix = np.zeros((N, sentences_per_pair))
        generated_sentences = {}
        for i, (word, _) in enumerate(closest_words):
            generated_sentences[word] = []
            for j in range(sentences_per_pair):
                latent_anchor = self.embed(anchor_word)
                latent_word = self.embed(word)
                mixed_latent = (latent_anchor + latent_word) / 2
                sentence = self.generate_from_latent(mixed_latent)
                complexity = self.semantic_complexity_metric(sentence)
                complexity_matrix[i, j] = complexity
                generated_sentences[word].append(sentence)
        return complexity_matrix, generated_sentences
