import os
import pandas as pd
import numpy as np
import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import defaultdict, Counter
#import textacy
#import spacy
from cleantext import clean

nltk.download('punkt_tab')
#spacy.cli.download("en_core_web_sm")
#nlp = spacy.load("en_core_web_sm")

class DataLoader:
    def __init__(self):
        self.frequent_tokens_ = set()

    def camel_to_words(self,text):
        # Convert CamelCase to words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Convert snake_case or snakyCamelCase to words
        text = re.sub(r'_([a-zA-Z])', r' \1', text)
        # Ensure spaces are properly added around each word
        return text
    
    def merge_datapoints_by_class(self, X_train, y_train):
        """
        Merges all text datapoints in X_train that belong to the same class in y_train.
        
        Parameters:
        - X_train: List of strings, where each string is a text data point.
        - y_train: List of class labels corresponding to each datapoint in X_train.
        
        Returns:
        - merged_X: List of merged text datapoints, one for each unique class in y_train.
        - merged_y: List of unique class labels corresponding to the merged datapoints.
        """
        # Dictionary to store merged data for each class
        class_to_text = defaultdict(list)
        
        # Iterate over X_train and y_train, appending the text to the appropriate class list
        for text, label in zip(X_train, y_train):
            class_to_text[label].append(text)
        
        # Merge text for each class and store in lists
        merged_X = [" ".join(texts) for texts in class_to_text.values()]
        merged_y = list(class_to_text.keys())
        
        return merged_X, merged_y

    def merge_text_rows(self, X_train, y_train):
        merged_X_train = []
        merged_y_train = []
        
        temp_text = ""
        temp_label = ""

        for i, text in enumerate(X_train):
            if temp_text:
                temp_text += " " + text
            else:
                temp_text = text
                temp_label = y_train[i]

            # Check if the current text ends with ':' or doesn't end with '.'
            if text.endswith(":") or text.endswith("]") or not text.endswith("."):
                continue  # Merge with the next line
            else:
                merged_X_train.append(temp_text)
                merged_y_train.append(temp_label)
                temp_text = ""  # Reset for the next text block

        # If any remaining text after loop, add it
        if temp_text:
            merged_X_train.append(temp_text)
            merged_y_train.append(temp_label)

        return np.array(merged_X_train), np.array(merged_y_train)

    def load_module_descriptions(self, directory, load_training_data_methods, merge_text):
        # Initialize lists to hold data from all methods
        all_descriptions = []
        all_labels = []
        
        for load_training_data_method in load_training_data_methods:
            if load_training_data_method == 'txt':
                descriptions = []
                labels = []
                # Ensure we're only processing folders directly under the specified directory
                for label_folder in os.listdir(directory):
                    label_path = os.path.join(directory, label_folder)

                    # Check if it's a directory
                    if os.path.isdir(label_path):
                        for root, _, files in os.walk(label_path):
                            # Skip directories that contain parentheses
                            if '(' in root or ')' in root:
                                continue

                            for file_name in files:
                                if file_name.endswith(".txt"):
                                    file_path = os.path.join(root, file_name)
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        description = f.read().strip()
                                        description = str(description)
                                        descriptions.append(description)
                                        labels.append(label_folder)  # Use the folder name as the label
                df = pd.DataFrame({'descriptions': descriptions, 'labels': labels})
                # Drop rows where descriptions are empty or contain only whitespace
                df = df[df['descriptions'].str.strip().astype(bool)]
                # Append results to all_descriptions and all_labels
                all_descriptions.extend(df['descriptions'].tolist())
                all_labels.extend(df['labels'].tolist())

            elif load_training_data_method == 'csv':
                combined_df = pd.DataFrame()

                # Loop through all CSV files in the directory
                for filename in os.listdir(directory):
                    if filename.endswith(".csv") and '(' not in filename and ')' not in filename:  # Process only CSV files
                        file_path = os.path.join(directory, filename)

                        # Load the current CSV file
                        df = pd.read_csv(file_path)

                        descriptions = df['Text'].tolist()
                        labels = df['Module'].tolist()
                        if merge_text:
                            descriptions, labels = self.merge_text_rows(descriptions, labels)

                        df = pd.DataFrame({'Text': descriptions,'Module': labels})

                        # Concatenate the current DataFrame with the combined DataFrame
                        combined_df = pd.concat([combined_df, df], ignore_index=True)

                # Append results to all_descriptions and all_labels
                all_descriptions.extend(combined_df['Text'].tolist())
                all_labels.extend(combined_df['Module'].tolist())

         # Final cleanup: keep only non-empty string pairs
        filtered_data = [
            (str(d).strip(), str(l).strip())
            for d, l in zip(all_descriptions, all_labels)
            if isinstance(d, str) and isinstance(l, str) and d.strip() and l.strip()
        ]
        all_descriptions, all_labels = zip(*filtered_data) if filtered_data else ([], [])
        processed_descriptions = [self.camel_to_words(text) for text in all_descriptions]

        return list(processed_descriptions), list(all_labels)
        

    
    def identify_frequent_tokens(self, X_train, y_train, threshold=0.75):
        """
        Identifies tokens that occur in more than `threshold` fraction of classes.

        Parameters:
        - X_train (list of str): Training texts.
        - y_train (list of str): Class labels for training texts.
        - threshold (float): Fraction of classes a token must appear in to be considered frequent.

        Sets:
        - self.frequent_tokens_: Set of tokens to be removed from all inputs.
        """
        classwise_tokens = defaultdict(set)

        # Collect tokens per class
        for text, label in zip(X_train, y_train):
            tokens = set(word_tokenize(text.lower()))
            classwise_tokens[label].update(tokens)

        # Invert: count how many classes each token appears in
        token_class_counts = Counter()
        for tokens in classwise_tokens.values():
            for token in tokens:
                token_class_counts[token] += 1

        num_classes = len(set(y_train))
        self.frequent_tokens_ = {
            token for token, count in token_class_counts.items()
            if count / num_classes >= threshold
        }
        
    def clean_text(self, text, rm_pct):
        cleaned = clean(text,
            fix_unicode=True,
            to_ascii=True,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=True,
            no_digits=True,
            no_currency_symbols=True,
            no_punct=rm_pct,
            replace_with_punct="",
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="",
            replace_with_currency_symbol=""
        )
        return cleaned


    def clean_text_data(self, X_train, y_train, remove_stopwords=True, rm_pct=False, remove_frequent=True):
        stop_words = set(stopwords.words('english')) if remove_stopwords else set()

        def clean_tokens(text):
            tokens = word_tokenize(text.lower())
            if rm_pct:
                tokens = [t for t in tokens if t.isalpha()]
            if remove_stopwords:
                tokens = [t for t in tokens if t not in stop_words]
            if remove_frequent and self.frequent_tokens_:
                tokens = [t for t in tokens if t not in self.frequent_tokens_]
            return " ".join(tokens)

        def rm_specials(text):
            cleanText = re.sub(r'[^a-zA-Z0-9.,;\'\s]', '', text)
            return cleanText
        
        cleaned_X = [self.clean_text(text, rm_pct) for text in X_train]
        cleaned_X = [clean_tokens(text) for text in cleaned_X]
        cleaned_X = [rm_specials(text) for text in cleaned_X]

        return np.array(cleaned_X), np.array(y_train)
    
    def filter_frequent_tokens_test_data(self, X_test, threshold=0.20):
        """
        Filters frequent tokens from the test data based on their frequency in the entire test set.
        Tokens that appear in more than `threshold` fraction of test samples will be removed.

        Parameters:
        - X_test (list of str): Test data (list of texts).
        - threshold (float): Fraction of test samples a token must appear in to be considered frequent.

        Returns:
        - cleaned_X (np.array): Test data with frequent tokens removed.
        """
        # Count token frequencies across all test points
        token_counts = Counter()
        for text in X_test:
            tokens = set(word_tokenize(text.lower()))  # Use set to avoid counting the same token multiple times per text
            token_counts.update(tokens)

        # Identify frequent tokens in test data based on their frequency across all texts
        total_docs = len(X_test)
        frequent_tokens_test = {token for token, count in token_counts.items() if count / total_docs >= threshold}

        # Clean the test data by filtering out frequent tokens
        def filter_tokens(text):
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t.isalpha()]
            tokens = [t for t in tokens if t not in frequent_tokens_test]
            return " ".join(tokens)

        # Apply filtering to all test data
        cleaned_X = [filter_tokens(text) for text in X_test]

        return np.array(cleaned_X)

    #def clean_with_textacy(text):
    #    doc = textacy.make_spacy_doc(text, lang=nlp)
    #    return " ".join(
    #        token.lemma_ for token in doc
    #        if not token.is_stop and not token.is_punct and token.is_alpha
    #    )

    #def minimal_clean(text):
    #    # Remove control characters and excessive whitespace
    #    text = re.sub(r"[\r\n\t]+", " ", text)
    #    text = re.sub(r"\s{2,}", " ", text)
    #    return text.strip()
    # def clean_training_data(self, X_train, y_train, chars_to_remove=None, remove_stopwords=True, apply_stemming=True, do_lemmatize=True, rm_pct=False):
    #     """
    #     Cleans the training data (X_train and y_train) by removing specified characters from the text,
    #     handling additional cleaning steps like stopword removal, lemmatization, punctuation removal, etc.

    #     Parameters:
    #     X_train (list of str): The training data containing text samples.
    #     y_train (list of str): The training labels.
    #     chars_to_remove (str): A string of characters to remove from the text (default includes common problematic ones).
    #     remove_stopwords (bool): Whether or not to remove stopwords (default is True).
    #     apply_stemming (bool): Whether or not to apply stemming (default is True).

    #     Returns:
    #     cleaned_X_train (list of str): Cleaned training data.
    #     cleaned_y_train (list of str): Cleaned labels.
    #     """
    #     stop_words = set(stopwords.words('english')) if remove_stopwords else None

    #     def rm_specials(text):
    #         if remove_stopwords:
    #             words = word_tokenize(text)
    #             words = [word for word in words if word not in stop_words]

    #         cleanText = re.sub(r'[^a-zA-Z0-9.,;\'\s]', '', text)
    #         return cleanText
        
    #     # Clean both X_train and y_train
    #     cleaned_X_train = [self.clean_text(text,rm_pct) for text in X_train]
    #     cleaned_X_train = [rm_specials(text) for text in cleaned_X_train]

    #     return np.array(cleaned_X_train), np.array(y_train)
    
    def apply_token_removal_to_test_data(self, X_test, remove_stopwords=True):
        """
        Applies the same token cleaning as training data using stored frequent tokens.
        """
        stop_words = set(stopwords.words('english')) if remove_stopwords else set()

        def clean_tokens(text):
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t.isalpha()]
            if remove_stopwords:
                tokens = [t for t in tokens if t not in stop_words]
            if self.frequent_tokens_:
                tokens = [t for t in tokens if t not in self.frequent_tokens_]
            return " ".join(tokens)

        cleaned_X = [self.clean_text(text, rm_pct=False) for text in X_test]
        cleaned_X = [clean_tokens(text) for text in cleaned_X]

        return np.array(cleaned_X)

    # def clean_training_data(self, X_train, y_train, chars_to_remove=None, remove_stopwords=True, apply_stemming=True, do_lemmatize=True):
    #     """
    #     Cleans the training data (X_train and y_train) by removing specified characters from the text,
    #     handling additional cleaning steps like stopword removal, lemmatization, punctuation removal, etc.

    #     Parameters:
    #     X_train (list of str): The training data containing text samples.
    #     y_train (list of str): The training labels.
    #     chars_to_remove (str): A string of characters to remove from the text (default includes common problematic ones).
    #     remove_stopwords (bool): Whether or not to remove stopwords (default is True).
    #     apply_stemming (bool): Whether or not to apply stemming (default is True).

    #     Returns:
    #     cleaned_X_train (list of str): Cleaned training data.
    #     cleaned_y_train (list of str): Cleaned labels.
    #     """
            
        
    #     # Define default characters to remove if not provided
    #     if chars_to_remove is None:
    #         chars_to_remove = "()[]{}<>/\\|;:\"'`~!@#$%^&*_=+-"
        
    #     # Compile a regular expression to match the characters to remove
    #     chars_pattern = re.compile(f"[{re.escape(chars_to_remove)}]")

    #     # Initialize stopwords and lemmatizer
    #     stop_words = set(stopwords.words('english')) if remove_stopwords else None
    #     lemmatizer = WordNetLemmatizer()
    #     stemmer = PorterStemmer() if apply_stemming else None

    #     def clean_text(text):

    #         text = self.camel_to_words(text)

    #         # Lowercase the text
    #         text = text.lower()

    #         # Remove URLs and email addresses
    #         text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    #         text = re.sub(r'\S+@\S+', '', text)

    #         # Remove numbers
    #         text = re.sub(r'\d+', ' ', text)

    #         # Replace `?` and `!` with `.`
    #         text = re.sub(r'[!?]', '.', text)

    #         # Remove all non-letter characters except ., ',', ';', and single quote
    #         text = re.sub(r'[^a-zA-Z0-9.,;\'\s]', '', text)

    #         # Remove punctuation
    #         #text = text.translate(str.maketrans('', '', string.punctuation))

    #         # Replace multiple dots with a single dot
    #         text = re.sub(r'\.+', '.', text)
    #         # Normalize accents
    #         text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    #         # Tokenize the text
    #         words = word_tokenize(text)

    #         # Remove stopwords if enabled
    #         if remove_stopwords:
    #             words = [word for word in words if word not in stop_words]

    #         # Lemmatization
    #         if do_lemmatize:
    #             words = [lemmatizer.lemmatize(word) for word in words]

    #         # Stemming if enabled
    #         if apply_stemming:
    #             words = [stemmer.stem(word) for word in words]

    #         # Explicitly remove spaces before punctuation
    #         text = re.sub(r' \.', '.', text)  # Replace " ." with "."
    #         text = re.sub(r' ,', ',', text)  # Replace " ," with ","
    #         # Join the words back into a single string
    #         text = ' '.join(words)

    #         # Replace ".." with "." and ". ." with "."
    #         #text = text.replace("..", ".").replace(". .", ".")
    #         #text = text.replace(" .", ".")
    #         #text = text.replace("more..", "")
    #         #text = text.replace(" .", " ")
    #         # Replace double whitespaces with single whitespace
    #         text = re.sub(r'\s+', ' ', text)
    #         return text

    #     # Clean both X_train and y_train
    #     cleaned_X_train = [clean_text(text) for text in X_train]

    #     return np.array(cleaned_X_train), np.array(y_train)