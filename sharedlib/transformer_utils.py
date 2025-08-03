from sklearn.base import BaseEstimator, TransformerMixin
import fasttext
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single column from a pandas DataFrame."""
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key]


class ImprovedFastTextVectorizer(BaseEstimator, TransformerMixin):
    """
    Improved FastText vectorizer that handles both utterances with various combination methods.
    Now gets fair comparison with TF-IDF by using both previous and current utterances.
    """
    def __init__(self, model_path, combination_method='concat'):
        self.model_path = model_path
        self.combination_method = combination_method  # 'concat', 'separate', 'average', 'current_only'
        self.model = None

    def __getstate__(self):
        """Handle pickling by excluding the unpicklable fasttext model."""
        state = self.__dict__.copy()
        del state['model']
        return state

    def __setstate__(self, state):
        """Handle unpickling by resetting model to None."""
        self.__dict__.update(state)
        self.model = None

    def _load_model_if_needed(self):
        """Loads the fasttext model into self.model if it's not already loaded."""
        if self.model is None:
            print(f"Loading FastText model from {self.model_path}...")
            self.model = fasttext.load_model(self.model_path)

    def fit(self, X, y=None):
        """Fit method loads the model to get dimensions."""
        self._load_model_if_needed()
        self.dim = self.model.get_dimension()
        return self

    def transform(self, X):
        """Transform text data into FastText embeddings using both utterances."""
        self._load_model_if_needed()
        embeddings = []
        
        for _, row in X.iterrows():
            prev_text = str(row['previous_utter_clean']).strip()
            curr_text = str(row['current_utter_clean']).strip()
            
            if self.combination_method == 'concat':
                # Concatenate previous and current utterances like TF-IDF does
                combined_text = f"{prev_text} [SEP] {curr_text}"
                embedding = self.model.get_sentence_vector(combined_text)
                
            elif self.combination_method == 'separate':
                # Generate separate embeddings then concatenate (double dimensions)
                prev_embedding = self.model.get_sentence_vector(prev_text)
                curr_embedding = self.model.get_sentence_vector(curr_text)
                embedding = np.concatenate([prev_embedding, curr_embedding])
                
            elif self.combination_method == 'average':
                # Average the embeddings from both utterances
                prev_embedding = self.model.get_sentence_vector(prev_text)
                curr_embedding = self.model.get_sentence_vector(curr_text)
                embedding = (prev_embedding + curr_embedding) / 2
                
            elif self.combination_method == 'current_only':
                # Original implementation for comparison (unfair but fast)
                embedding = self.model.get_sentence_vector(curr_text)
                
            else:
                raise ValueError(f"Unknown combination method: {self.combination_method}")
            
            embeddings.append(embedding)
        
        return np.array(embeddings)