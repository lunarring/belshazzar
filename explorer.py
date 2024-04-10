import os
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from gensim.models import KeyedVectors
import numpy as np

class BottleneckT5Autoencoder:
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval()

    def embed(self, text: str) -> torch.FloatTensor:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        decoder_inputs = self.tokenizer('', return_tensors='pt').to(self.device)
        return self.model(**inputs, decoder_input_ids=decoder_inputs['input_ids'], encode_only=True)[0]

    def generate_from_latent(self, latent: torch.FloatTensor, max_length=512, temperature=1.0) -> str:
        dummy_text = '.'
        dummy = self.embed(dummy_text)
        perturb_vector = latent - dummy
        self.model.perturb_vector = perturb_vector
        input_ids = self.tokenizer(dummy_text, return_tensors='pt').to(self.device).input_ids
        output = self.model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, temperature=temperature, top_p=0.9, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class TextAnalysis:
    def __init__(self, gpt_model_name='gpt2', word2vec_model_path, t5_model_path='thesephist/contra-bottleneck-t5-large-wikipedia'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.gpt_model = GPT2Model.from_pretrained(gpt_model_name).to(self.device)
        self.word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
        self.autoencoder = BottleneckT5Autoencoder(t5_model_path, self.device)

    def semantic_complexity_metric(self, text):
        inputs = self.gpt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.gpt_model(**inputs)
        return torch.std(outputs.last_hidden_state, dim=1).mean().item()

    def closest_words_word2vec(self, word, N=5):
        return self.word2vec_model.most_similar(positive=[word], topn=N)

    def generate_and_analyze_complexity(self, anchor_word, N=5, sentences_per_pair=10):
        closest_words = self.closest_words_word2vec(anchor_word, N)
        complexity_matrix = np.zeros((N, sentences_per_pair))
        generated_sentences = {}
        for i, (word, _) in enumerate(closest_words):
            generated_sentences[word] = []
            for j in range(sentences_per_pair):
                latent_anchor = self.autoencoder.embed(anchor_word)
                latent_word = self.autoencoder.embed(word)
                mixed_latent = (latent_anchor + latent_word) / 2
                sentence = self.autoencoder.generate_from_latent(mixed_latent)
                complexity = self.semantic_complexity_metric(sentence)
                complexity_matrix[i, j] = complexity
                generated_sentences[word].append(sentence)
        return complexity_matrix, generated_sentences
