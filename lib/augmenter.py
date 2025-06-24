import random
import nltk
from rake_nltk import Rake
import yake
from transformers import pipeline, BartTokenizer
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from imblearn.over_sampling import SMOTE
import numpy as np
import os
import openai
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import math
config = {
    'OPENAI_API_BASE': ## TODO ADD,
    'OPENAI_API_KEY': ## TODO ADD,
    'OPENAI_API_VERSION': ## TODO ADD,
    'EMBEDDING_MODEL': ## TODO ADD
}

# Set up the OpenAI API client for Azure
#openai.api_type = "azure"
openai.azure_endpoint = config['OPENAI_API_BASE']
openai.api_key = config['OPENAI_API_KEY']
openai.api_version = config['OPENAI_API_VERSION']


class Augmenter:
    def __init__(self):
        self.rake = Rake(include_repeated_phrases=False)
        self.tokenizer = BartTokenizer.from_pretrained('./huggingface_models/bart-large-cnn/')
        self.summarizer = pipeline("summarization", model="./huggingface_models/bart-large-cnn/")
        #self.aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_p=0.1)

    def augment_text(self, descriptions, labels, augmenting_methods=['sentences'], sliding_window_sizes=None, sliding_window_overlap=False, use_resampling=False, upsample_largest_class=None):
        augmented_descriptions = []
        augmented_labels = []

        # Collect class-wise descriptions and labels
        class_data = {label: [] for label in set(labels)}
        for description, label in zip(descriptions, labels):
            class_data[label].append(description)

        # Extract paragraphs (formerly "file_content") only once
        all_paragraphs = {label: self.extract_paragraphs(class_data[label], augmenting_methods) for label in class_data}

        # Upsample largest class if the parameter is set
        max_class_size = max(len(class_data[label]) for label in class_data)
        if upsample_largest_class:
            target_size = int(max_class_size * upsample_largest_class)
        else:
            target_size = max_class_size

        # Apply augmentation and balance class sizes
        for label, data in class_data.items():
            if not data:
                print(f"No data available for label '{label}'. Skipping augmentation.")
                continue

            if use_resampling:
                # Augment the smaller classes to match target size
                augmented_samples = self.augment_class_to_size(
                    all_paragraphs[label], label, target_size, augmenting_methods,
                    sliding_window_sizes, sliding_window_overlap
                )
            else:
                # Augment each description in the class individually
                augmented_samples = []
                for description in all_paragraphs[label]:
                    augmented_samples.extend(
                        self.apply_augmentation(
                            description, augmenting_methods,
                            sliding_window_sizes, sliding_window_overlap
                        )
                    )

                augmented_descriptions.extend(augmented_samples)
                augmented_labels.extend([label] * len(augmented_samples))

        augmented_descriptions = np.array(augmented_descriptions)
        augmented_labels = np.array(augmented_labels)

        return augmented_descriptions, augmented_labels

    def extract_paragraphs(self, descriptions, augmenting_methods):
        """
        Extract paragraphs (formerly "file_content") only once.
        """
        paragraphs = []
        for description in descriptions:
            for method in augmenting_methods:
                if method == 'file_content' or method == 'paragraphs':  # Handle the old "file_content"
                    paragraphs.append(description)
        return paragraphs

    def augment_class_to_size(self, data, label, target_size, augmenting_methods, eda_num_aug, paraphrase_num_return_sequences, sliding_window_sizes, sliding_window_overlap):
        augmented_samples = []
        while len(augmented_samples) < target_size:
            for description in data:
                augmented_batch = self.apply_augmentation(description, augmenting_methods, eda_num_aug, paraphrase_num_return_sequences, sliding_window_sizes, sliding_window_overlap)
                augmented_samples.extend(augmented_batch)
                if len(augmented_samples) >= target_size:
                    break
        return augmented_samples[:target_size]

    def apply_augmentation(self, description, augmenting_methods, sliding_window_sizes, sliding_window_overlap):
        augmented_samples = []

        # **Include the original description** in the augmented samples
        augmented_samples.append(description)

        sentence_count = len(sent_tokenize(description))

        # Sentence-level augmentations
        
        sentences = []
        for method in augmenting_methods:
            if method == 'sentences' and sentence_count > 1:
                sentences.extend(sent_tokenize(description))
            elif method == 'random_sentence_augment':
                sentences.extend(self.random_sentence_augment(description))
            elif method == 'sentence_shuffling':
                sentences.extend(self.shuffle_sentences(description))
            elif method == 'abstractive_summarization_augment':
                if len(description.split()) > 10:
                    sentences.extend(self.abstractive_summarization_augment(description))
            elif method == 'sentences_sliding_window' and sentence_count > 2:
                sentences.extend(self.sentences_sliding_window(description))
            elif method == 'stopwords_sliding_window':
                sentences.extend(self.stopwords_sliding_window(description))
            elif method == 'decompose_text_with_stopwords':
                sentences.extend(self.decompose_text_with_stopwords(description))
            
                

        # Keyword generation
        keywords = []
        for method in augmenting_methods:
            if method == 'rake':
                #if len(description.split()) > 30:
                keywords.extend(self.extract_keywords_rake(description))
            elif method == 'yake':
                keywords.extend(self.extract_keywords_yake(description))

        # Word-level augmentation
        aug_text = []
        for method in augmenting_methods:
            if method == 'synonym_augment':
                aug_text.extend(self.synonym_augment(description))
            elif method == 'random_word_augment':
                aug_text.extend(self.random_word_augment(description))
            elif method == 'sentence_spelling_augment':
                aug_text.extend(self.sentence_spelling_augment(description))
            elif method == 'sentence_back_translation_augment':
                aug_text.extend(self.sentence_back_translation_augment(description))

        augmented_samples.extend(sentences)
        augmented_samples.extend(keywords)
        augmented_samples.extend(aug_text)

        return augmented_samples


    def shuffle_sentences(self, text):
        random.seed(42)  # Ensure reproducibility
        sentences = sent_tokenize(text)
        sent_len = len(sentences)
        shuffled_texts = []
        if sent_len > 1:  # Ensure there's more than one sentence to shuffle
            possible_permutations = math.factorial(sent_len)
            max_shuffles = min(possible_permutations, 5)
            # Perform shuffle the specified number of times
            for _ in range(max_shuffles):
                random.shuffle(sentences)  # Shuffle in-place
                shuffled_texts.append(' '.join(sentences))  # Join sentences back into a single string

        return shuffled_texts

    def extract_keywords_rake(self, text):
        n_keyphrases = int(len(text.split())/10) # one keyphrase per 10 words
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()[0:n_keyphrases]

    def extract_keywords_yake(self, text, deduplication_threshold=0.9, max_ngram_size=5, num_keywords=3):
        kw_extractor = yake.KeywordExtractor(lan='en',
                                             dedupLim=deduplication_threshold,
                                             n=max_ngram_size,
                                             top=num_keywords)
        keywords = kw_extractor.extract_keywords(text)
        return [keyword for keyword, score in keywords]
    
    def random_sentence_augment(self,text):
        aug = nas.RandomSentAug()
        sentences = sent_tokenize(text)
        sent_len = len(sentences)
        shuffled_texts = []
        if sent_len > 1:  # Ensure there's more than one sentence to shuffle
            possible_permutations = math.factorial(sent_len)
            max_shuffles = min(possible_permutations, 5)
            for _ in range(max_shuffles):
                aug_texts = aug.augment(text)
                shuffled_texts.append(' '.join(aug_texts))  # Join sentences back into a single string

        return shuffled_texts
    

    def abstractive_summarization_augment(self,text):
        # Tokenize and truncate the input text to max length of 1024 tokens
        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors='pt')

        
        # Decode the truncated input back into text form
        truncated_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        text_length = len(truncated_text.split())
        # Set minimum length first to ensure max_length is not less than min_length
        # Ensure max_length and min_length are within bounds
        max_length = min(text_length, 1024)  # Cannot exceed 1024 tokens
        min_length = max(1, int(max_length * 0.75))  # At least 30 tokens, or 75% of max_length
        
        # Perform summarization on the truncated text
        # Pass the input as a list to maintain batch consistency
        summarization_result = self.summarizer(
            [truncated_text],  # Ensure input is a list
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        # Extract the summary text
        aug_text = summarization_result[0]['summary_text']
        
        return aug_text
    
    #aug = nas.AbstSummAug(model_path = "./huggingface_models/bart-large-cnn/", tokenizer_path = "./huggingface_models/bart-large-cnn/",max_length=1024)
        #aug_texts = aug.augment(text)
    def synonym_augment(self, text):
        aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_p=0.1)
        aug_texts = []#aug.augment(text)
        for _ in range(4):
            augmented = aug.augment(text)
            if isinstance(augmented, list):
                augmented = ' '.join(augmented)
            aug_texts.append(augmented)
        return aug_texts
    
    def random_word_augment(self,text):
        aug = naw.RandomWordAug(action='substitute')
        aug_texts = []
        for _ in range(4):
            augmented = aug.augment(text)
            if isinstance(augmented, list):
                augmented = ' '.join(augmented)
            aug_texts.append(augmented)
        return aug_texts
    
    def sentence_spelling_augment(self,text):
        aug = naw.SpellingAug()
        aug_texts = []#aug.augment(text)
        for _ in range(4):
            augmented = aug.augment(text)
            if isinstance(augmented, list):
                augmented = ' '.join(augmented)
            aug_texts.append(augmented)
        return aug_texts
    
    def sentence_back_translation_augment(self,text):
        aug = naw.BackTranslationAug(from_model_name='./huggingface_models/opus-mt-en-de', to_model_name='./huggingface_models/opus-mt-de-en')
        aug_texts = aug.augment(text)
        return aug_texts
    
    def sentences_sliding_window(self,text,window_size=3):
        sentences = sent_tokenize(text)
        return [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]

    def stopwords_sliding_window(self,text,window_size=3):
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
        # Filter out stop words from each sentence
        filtered_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            filtered_sentence = ' '.join([word for word in words if word.lower() not in stop_words])
            filtered_sentences.append(filtered_sentence)

        # Create data points using a sliding window approach
        data_points = [' '.join(filtered_sentences[i:i + window_size]) for i in range(len(filtered_sentences) - window_size + 1)]
    
        return data_points
    
    def decompose_text_with_stopwords(self,text):
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
        all_segments = []

        # Iterate over each sentence and decompose it based on stop words
        for sentence in sentences:
            words = word_tokenize(sentence)
            current_segment = []
            
            # Split each sentence into segments based on stop words
            for word in words:
                if word.lower() in stop_words:
                    if current_segment:  # If we have words collected in the segment, save it
                        all_segments.append(' '.join(current_segment))
                        current_segment = []  # Reset for the next segment
                else:
                    current_segment.append(word)  # Keep adding words to the current segment
            
            # Add the last segment of the sentence if there are any words left
            if current_segment:
                all_segments.append(' '.join(current_segment))
        return all_segments
    
    def smote_resample(self, X_train_embeddings, y_train, n_neighbors=5):
        class_counts = Counter(y_train)
        smallest_class_size = min(class_counts.values())
        
        # Determine n_neighbors based on the provided option
        if n_neighbors == 'sqrt_min_class':
            computed_n_neighbors = max(1, int(math.sqrt(smallest_class_size)))
        elif n_neighbors == 'min_class':
            computed_n_neighbors = max(1, smallest_class_size - 1)
        else:
            # Assume n_neighbors is an integer
            computed_n_neighbors = n_neighbors
        
        # Ensure that n_neighbors does not exceed smallest_class_size - 1
        max_allowed_neighbors = smallest_class_size - 1
        if max_allowed_neighbors < 1:
            raise ValueError(
                f"Cannot apply SMOTE: the smallest class has only {smallest_class_size} samples, "
                f"which is less than required for SMOTE."
            )
        
        # Set n_neighbors to the minimum of computed_n_neighbors and max_allowed_neighbors
        final_n_neighbors = min(computed_n_neighbors, max_allowed_neighbors)
        
        # Optionally, you can log or print the chosen n_neighbors
        print(
            f"Using n_neighbors={final_n_neighbors} for SMOTE based on smallest class size={smallest_class_size}"
        )
        
        # Initialize SMOTE with the final n_neighbors
        self.smote = SMOTE(k_neighbors=final_n_neighbors)
        
        # Perform resampling
        X_train_embeddings_resampled, y_train_resampled = self.smote.fit_resample(
            X_train_embeddings, y_train
        )
        
        return X_train_embeddings_resampled, y_train_resampled