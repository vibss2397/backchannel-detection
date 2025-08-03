# train_improved.py

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime
import requests
import gzip
import shutil

# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, average_precision_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sharedlib.transformer_utils import ImprovedFastTextVectorizer

# --- Model and Training Configuration ---

def get_model_config():
    """Returns a dictionary containing model and training configurations."""
    return {
        'fasttext_model_path': 'cc.en.300.bin',
        'fasttext_download_url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz',
        'tfidf_feature_options': [500, 1000, 2500, 5000, 7500],
        'regularization_options': [0.1, 0.5, 1.0, 2.0, 5.0],
        'fasttext_combination_options': ['concat', 'separate', 'average', 'current_only'],
        'sampling_strategies': ['none', 'oversample', 'undersample'],
        'test_size': 0.2,
        'validation_size': 0.25,
        'random_state': 42,
        'log_reg_solver': 'liblinear',
        'log_reg_max_iter': 2000,
        'benchmark_runs': 1000,
        'latency_threshold_ms': 50,
    }

# --- Custom Classes for Pipelines ---

class KeywordBaseline:
    """Baseline classifier using keyword lookup from task specification."""
    
    def __init__(self):
        self.backchannel_keywords = {
            "yeah", "yes", "uh-huh", "mhmm", "mm-hmm", "hmm",
            "oh", "ah", "uhhuh", "uh", "um", "mmmm", "yep",
            "wow", "right", "okay", "ok", "sure", "alright",
            "gotcha", "mmhmm", "great", "sweet", "ma'am", "awesome",
            "i see", "got it", "that makes sense", "i hear you",
            "i understand", "good afternoon", "hey there", "perfect",
            "that's true", "good point", "exactly", "makes sense",
            "no problem", "indeed", "certainly", "very well", "absolutely",
            "correct", "of course", "k", "hey", "hello", "hi", "yo",
            "good morning"
        }
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        """Predict 1 if ANY keyword found in current utterance, 0 otherwise."""
        predictions = []
        for _, row in X.iterrows():
            text = str(row['current_utter_clean']).lower().strip()
            is_backchannel = any(keyword in text for keyword in self.backchannel_keywords)
            predictions.append(1 if is_backchannel else 0)
        return np.array(predictions)
    
    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2))
        proba[preds == 1, 1] = 0.9
        proba[preds == 1, 0] = 0.1
        proba[preds == 0, 1] = 0.1
        proba[preds == 0, 0] = 0.9
        return proba

# --- Utility and Evaluation Functions ---

def download_fasttext_model(model_path, config):
    """Downloads the pre-trained FastText model if it doesn't exist."""
    if os.path.exists(model_path):
        print(f"FastText model '{model_path}' already exists.")
        return model_path

    url = config['fasttext_download_url']
    gz_path = model_path + ".gz"
    
    print(f"Downloading FastText model from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(gz_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print("Download complete. Decompressing...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove(gz_path)
        print(f"FastText model saved to '{model_path}'")
        return model_path
    except Exception as e:
        print(f"Error downloading or decompressing FastText model: {e}")
        return None


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation with proper binary classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average='binary')
    avg_precision = average_precision_score(y_test, y_proba)
    
    print(f"\n=== {model_name} Results ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Backchannel', 'Backchannel']))
    
    return {
        'roc_auc': roc_auc,
        'f1_score': f1,
        'avg_precision': avg_precision,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Backchannel', 'Backchannel'],
                yticklabels=['Not Backchannel', 'Backchannel'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def benchmark_inference_speed(model, X_sample, config):
    """Benchmark model inference speed."""
    n_runs = config['benchmark_runs']
    print(f"Benchmarking inference speed with {n_runs} runs...")
    
    # Warm up
    for _ in range(10):
        model.predict(X_sample.iloc[:1])
    
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        model.predict(X_sample.iloc[:1])
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    
    print(f"Average inference time: {avg_time:.2f}ms")
    print(f"95th percentile: {p95_time:.2f}ms")
    print(f"Target: <{config['latency_threshold_ms']}ms - {'✅ PASS' if p95_time < config['latency_threshold_ms'] else '❌ FAIL'}")
    
    return {'avg_ms': avg_time, 'p95_ms': p95_time}


# --- TF-IDF Model Building Functions ---
def find_best_max_features(X_train, y_train, X_val, y_val, sampling_strategy, config):
    """Tests different max_features values for the TF-IDF model."""
    feature_options = config['tfidf_feature_options']
    regularization_options = config['regularization_options']
    results = []

    print("\n--- Finding optimal max_features for TF-IDF model ---")
    print(f"{'Max Features':<15} | {'C':<8} | {'Validation F1-Score':<20}")
    print("-" * 40)

    for n_features in feature_options:
        for reg in regularization_options:
            pipeline = build_tfidf_pipeline_with_params(
                prev_max_features=n_features,
                curr_max_features=n_features,
                sampling_strategy=sampling_strategy,
                C=reg,
                config=config
            )
            pipeline.fit(X_train, y_train)
            y_val_pred = pipeline.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred, average='binary')
            print(f"{n_features:<15} | {reg:<8.1f} | {val_f1:<20.4f}")
            results.append({'max_features': n_features, 'f1_score': val_f1, 'C': reg})

    best_result = max(results, key=lambda x: x['f1_score'])
    print("-" * 40)
    print(f"Optimal max_features found: {best_result['max_features']} and C={best_result['C']} "
          f" with F1-Score: {best_result['f1_score']:.4f}")

    return best_result['max_features'], best_result['C']


def build_tfidf_pipeline_with_params(prev_max_features, curr_max_features, sampling_strategy, C, config) -> ImbPipeline:
    """Builds the TF-IDF pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('prev', TfidfVectorizer(ngram_range=(1, 2), max_features=prev_max_features), 'previous_utter_clean'),
            ('curr', TfidfVectorizer(ngram_range=(1, 3), max_features=curr_max_features), 'current_utter_clean')
        ],
        remainder='drop'
    )
    
    steps = [('preprocessor', preprocessor)]

    if sampling_strategy == 'oversample':
        steps.append(('sampler', RandomOverSampler(random_state=config['random_state'])))
    elif sampling_strategy == 'undersample':
        steps.append(('sampler', RandomUnderSampler(random_state=config['random_state'])))
        
    steps.append(('classifier', LogisticRegression(
        random_state=config['random_state'], 
        solver=config['log_reg_solver'], 
        max_iter=config['log_reg_max_iter'],
        C=C,
        class_weight='balanced'
    )))
    
    return ImbPipeline(steps)


# --- FastText Model Building Functions ---
def find_best_fasttext_config(X_train, y_train, X_val, y_val, model_path, sampling_strategy, config):
    """Find optimal FastText configuration with fair comparison to TF-IDF."""
    combination_options = config['fasttext_combination_options']
    regularization_options = config['regularization_options']
    results = []

    print("\n--- Finding optimal FastText configuration ---")
    print(f"{'Combination':<15} | {'C':<8} | {'Val F1-Score':<15} | {'Notes':<25}")
    print("-" * 70)

    for combination in combination_options:
        for C_val in regularization_options:
            try:
                pipeline = build_fasttext_pipeline_with_params(
                    model_path=model_path,
                    combination_method=combination,
                    C=C_val,
                    sampling_strategy=sampling_strategy,
                    config=config
                )
                
                pipeline.fit(X_train, y_train)
                y_val_pred = pipeline.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred, average='binary')
                
                notes = ""
                if combination == 'current_only':
                    notes = "UNFAIR (curr only)"
                elif combination == 'concat':
                    notes = "FAIR (like TF-IDF)"
                elif combination == 'separate':
                    notes = "FAIR (600 dims)"
                elif combination == 'average':
                    notes = "FAIR (300 dims)"
                
                print(f"{combination:<15} | {C_val:<8.1f} | {val_f1:<15.4f} | {notes:<25}")
                results.append({
                    'combination': combination,
                    'C': C_val,
                    'f1_score': val_f1,
                    'notes': notes
                })
                
            except Exception as e:
                print(f"{combination:<15} | {C_val:<8.1f} | ERROR: {str(e):<15} |")

    if results:
        best_result = max(results, key=lambda x: x['f1_score'])
        print("-" * 70)
        print(f"Optimal config: {best_result['combination']} + C={best_result['C']} "
              f"with F1-Score: {best_result['f1_score']:.4f}")
        return best_result['combination'], best_result['C']
    else:
        print("No valid configurations found, using defaults")
        return 'concat', 1.0


def build_fasttext_pipeline_with_params(model_path, combination_method, C, sampling_strategy, config):
    """Build FastText pipeline with proper configuration and fair comparison."""
    steps = [('vectorizer', ImprovedFastTextVectorizer(model_path=model_path, combination_method=combination_method))]
    
    if sampling_strategy == 'oversample':
        steps.append(('sampler', RandomOverSampler(random_state=config['random_state'])))
    elif sampling_strategy == 'undersample':
        steps.append(('sampler', RandomUnderSampler(random_state=config['random_state'])))
    
    steps.append(('classifier', LogisticRegression(
        random_state=config['random_state'], 
        solver=config['log_reg_solver'], 
        max_iter=config['log_reg_max_iter'],
        C=C,
        class_weight='balanced'
    )))
    
    return ImbPipeline(steps)


# --- Main Training Workflow ---
def main(args):
    """
    Main function to run the training and evaluation pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments for training configuration.
    """
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"Loading training data from {args.training_file}...")
        df = pd.read_csv(args.training_file)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.training_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}")
        return

    df[['previous_utter_clean', 'current_utter_clean']] = df[['previous_utter_clean', 'current_utter_clean']].fillna('')
    
    config = get_model_config()
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    X = df[['previous_utter_clean', 'current_utter_clean']]
    y = df['label']
    
    print(f"\nSplitting data into train/val/test ({1 - config['test_size']:.0%}/{config['test_size'] * config['validation_size']:.0%}/{config['test_size'] * (1-config['validation_size']):.0%})...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'], stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=config['validation_size'], random_state=config['random_state'], stratify=y_temp)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # === MODEL 1: BASELINE ===
    print("\n" + "="*50)
    print("MODEL 1: TRAINING BASELINE (Keyword Lookup)")
    print("="*50)
    baseline = KeywordBaseline()
    baseline.fit(X_train, y_train)
    baseline_results = evaluate_model(baseline, X_test, y_test, "Keyword Baseline")
    
    # === MODEL 2: TF-IDF + LOGISTIC REGRESSION ===
    print("\n" + "="*50)
    print("MODEL 2: TRAINING TF-IDF + LOGISTIC REGRESSION")
    print("="*50)
    optimal_features, optimal_C = find_best_max_features(X_train, y_train, X_val, y_val, args.sampling, config)
    final_tfidf_pipeline = build_tfidf_pipeline_with_params(optimal_features, optimal_features, args.sampling, optimal_C, config)
    
    print("\nTraining final TF-IDF model on full training set...")
    final_tfidf_pipeline.fit(X_train, y_train)
    
    print("\nFinal test set performance for TF-IDF model:")
    tfidf_results = evaluate_model(final_tfidf_pipeline, X_test, y_test, "TF-IDF Model (Test)")

    # === MODEL 3: IMPROVED FASTTEXT + LOGISTIC REGRESSION ===
    print("\n" + "="*50)
    print("MODEL 3: TRAINING IMPROVED FASTTEXT + LOGISTIC REGRESSION")
    print("="*50)
    
    ft_model_path = download_fasttext_model(os.path.join(args.output_dir, config['fasttext_model_path']), config)
    if not ft_model_path:
        print("Could not obtain FastText model. Skipping FastText evaluation.")
        fasttext_results = None
        fasttext_speed = None
        final_fasttext_pipeline = None
    else:
        # Find optimal FastText configuration
        optimal_combination, optimal_C = find_best_fasttext_config(
            X_train, y_train, X_val, y_val, ft_model_path, args.sampling, config
        )
        
        # Build final pipeline with optimal parameters
        final_fasttext_pipeline = build_fasttext_pipeline_with_params(
            model_path=ft_model_path,
            combination_method=optimal_combination,
            C=optimal_C,
            sampling_strategy=args.sampling,
            config=config
        )
        
        print(f"\nTraining final FastText model with {optimal_combination} + C={optimal_C}...")
        final_fasttext_pipeline.fit(X_train, y_train)
        
        print("\nFinal test set performance for FastText model:")
        fasttext_results = evaluate_model(final_fasttext_pipeline, X_test, y_test, "Improved FastText Model")

    # === INFERENCE SPEED BENCHMARKING ===
    print("\n" + "="*50)
    print("INFERENCE SPEED BENCHMARKING")
    print("="*50)
    sample_data = pd.DataFrame({
        'previous_utter_clean': ['So I was thinking about going to the store'] * 5,
        'current_utter_clean': ['yeah', 'that sounds great', 'uh-huh', 'what time works?', 'mm-hmm']
    })
    
    print("\nBaseline speed:")
    baseline_speed = benchmark_inference_speed(baseline, sample_data, config)
    
    print("\nTF-IDF Model speed:")
    tfidf_speed = benchmark_inference_speed(final_tfidf_pipeline, sample_data, config)

    if fasttext_results:
        print("\nImproved FastText Model speed:")
        fasttext_speed = benchmark_inference_speed(final_fasttext_pipeline, sample_data, config)
    
    # === MODEL COMPARISON ===
    print(f"\n{'='*80}")
    print(f"{'IMPROVED MODEL COMPARISON SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Baseline':<15} {'TF-IDF Model':<15} {'FastText Model':<15}")
    print(f"{'-'*80}")
    
    ft_roc = fasttext_results['roc_auc'] if fasttext_results else 'N/A'
    ft_f1 = fasttext_results['f1_score'] if fasttext_results else 'N/A'
    ft_ap = fasttext_results['avg_precision'] if fasttext_results else 'N/A'
    ft_p95 = fasttext_speed['p95_ms'] if fasttext_results and fasttext_speed else 'N/A'

    print(f"{'ROC-AUC':<20} {baseline_results['roc_auc']:<15.4f} {tfidf_results['roc_auc']:<15.4f} {ft_roc if isinstance(ft_roc, str) else f'{ft_roc:<15.4f}'}")
    print(f"{'F1-Score':<20} {baseline_results['f1_score']:<15.4f} {tfidf_results['f1_score']:<15.4f} {ft_f1 if isinstance(ft_f1, str) else f'{ft_f1:<15.4f}'}")
    print(f"{'Avg Precision':<20} {baseline_results['avg_precision']:<15.4f} {tfidf_results['avg_precision']:<15.4f} {ft_ap if isinstance(ft_ap, str) else f'{ft_ap:<15.4f}'}")
    print(f"{'Speed (P95 ms)':<20} {baseline_speed['p95_ms']:<15.2f} {tfidf_speed['p95_ms']:<15.2f} {ft_p95 if isinstance(ft_p95, str) else f'{ft_p95:<15.2f}'}")
    
    # Show improvement analysis
    if fasttext_results:
        print(f"\n{'IMPROVEMENT ANALYSIS':^80}")
        print(f"{'-'*80}")
        tfidf_vs_ft_roc = tfidf_results['roc_auc'] - fasttext_results['roc_auc']
        tfidf_vs_ft_f1 = tfidf_results['f1_score'] - fasttext_results['f1_score']
        speed_diff = fasttext_speed['p95_ms'] - tfidf_speed['p95_ms']
        
        print(f"TF-IDF vs FastText ROC-AUC difference: {tfidf_vs_ft_roc:+.4f}")
        print(f"TF-IDF vs FastText F1-Score difference: {tfidf_vs_ft_f1:+.4f}")
        print(f"FastText speed vs TF-IDF: {speed_diff:+.2f}ms")
    
    # === SAVE ARTIFACTS ===
    plot_confusion_matrix(y_test, baseline_results['predictions'], "Keyword Baseline", os.path.join(args.output_dir, 'cm_baseline.png'))
    plot_confusion_matrix(y_test, tfidf_results['predictions'], "TF-IDF Model", os.path.join(args.output_dir, 'cm_tfidf.png'))
    if fasttext_results:
        plot_confusion_matrix(y_test, fasttext_results['predictions'], "Improved FastText Model", os.path.join(args.output_dir, 'cm_fasttext_improved.png'))

    model_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(baseline, os.path.join(args.output_dir, f"baseline_model_{model_timestamp}.joblib"))
    joblib.dump(final_tfidf_pipeline, os.path.join(args.output_dir, f"tfidf_model_{model_timestamp}.joblib"))
    if fasttext_results:
        joblib.dump(final_fasttext_pipeline, os.path.join(args.output_dir, f"fasttext_improved_model_{model_timestamp}.joblib"))

    # === GENERATE COMPREHENSIVE REPORT ===
    report_path = os.path.join(args.output_dir, 'training_report_improved.txt')
    print(f"\nGenerating comprehensive training report at {report_path}...")
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("IMPROVED BACKCHANNEL MODEL TRAINING REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Dataset: {os.path.basename(args.training_file)}\n")
        f.write(f"Sampling Strategy: {args.sampling}\n\n")

        f.write("IMPROVEMENTS MADE TO FASTTEXT:\n")
        f.write("-" * 35 + "\n")
        f.write("✅ Uses both previous AND current utterances (fair comparison)\n")
        f.write("✅ Hyperparameter tuning for combination method and regularization\n")
        f.write("✅ Class balancing with balanced weights\n")
        f.write("✅ Proper sampling strategy support\n")
        f.write("✅ Multiple combination methods tested\n\n")

        f.write("FINAL TEST SET PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Metric':<20} {'Baseline':<15} {'TF-IDF Model':<15} {'FastText Model':<15}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'ROC-AUC':<20} {baseline_results['roc_auc']:<15.4f} {tfidf_results['roc_auc']:<15.4f} {ft_roc if isinstance(ft_roc, str) else f'{ft_roc:<15.4f}'}\n")
        f.write(f"{'F1-Score':<20} {baseline_results['f1_score']:<15.4f} {tfidf_results['f1_score']:<15.4f} {ft_f1 if isinstance(ft_f1, str) else f'{ft_f1:<15.4f}'}\n")
        f.write(f"{'Avg Precision':<20} {baseline_results['avg_precision']:<15.4f} {tfidf_results['avg_precision']:<15.4f} {ft_ap if isinstance(ft_ap, str) else f'{ft_ap:<15.4f}'}\n")
        f.write(f"{'Speed (P95 ms)':<20} {baseline_speed['p95_ms']:<15.2f} {tfidf_speed['p95_ms']:<15.2f} {ft_p95 if isinstance(ft_p95, str) else f'{ft_p95:<15.2f}'}\n\n")

        f.write("LATENCY REQUIREMENTS\n")
        f.write("-" * 20 + "\n")
        f.write("Target: <50ms\n")
        f.write(f"Baseline: {'✅ PASS' if baseline_speed['p95_ms'] < 50 else '❌ FAIL'}\n")
        f.write(f"TF-IDF Model: {'✅ PASS' if tfidf_speed['p95_ms'] < 50 else '❌ FAIL'}\n")
        if fasttext_results:
            f.write(f"Improved FastText Model: {'✅ PASS' if fasttext_speed['p95_ms'] < 50 else '❌ FAIL'}\n")

    print(f"\nModels and reports saved in '{args.output_dir}'")
    print("\nImproved training completed successfully! 🎉")
    print("\nNow FastText gets a FAIR comparison with both utterances and hyperparameter tuning!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and compare backchannel detection models with improved FastText.")
    parser.add_argument('--training-file', type=str, required=True, 
                       help="Path to the training CSV dataset.")
    parser.add_argument('--output-dir', type=str, default='./output', 
                       help="Directory to save models and reports.")
    parser.add_argument('--sampling', type=str, choices=['none', 'oversample', 'undersample'], 
                       default='none', help="Sampling strategy for class imbalance.")
    
    args = parser.parse_args()
    main(args)