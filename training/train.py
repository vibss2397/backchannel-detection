# train.py

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

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
            text = str(row['current_clean']).lower().strip()
            # Liberal baseline: any keyword presence = backchannel
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

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation with proper binary classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
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

def benchmark_inference_speed(model, X_sample, n_runs=1000):
    """Benchmark model inference speed."""
    print(f"Benchmarking inference speed with {n_runs} runs...")
    
    # Warm up
    for _ in range(10):
        model.predict(X_sample.iloc[:1])
    
    # Benchmark
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
    print(f"Target: <50ms - {'✅ PASS' if p95_time < 50 else '❌ FAIL'}")
    
    return {'avg_ms': avg_time, 'p95_ms': p95_time}


def find_best_max_features(X_train, y_train, X_val, y_val, sampling_strategy):
    """
    Tests different max_features values to find the best one based on validation F1-score.
    """
    # Define a range of values to test
    feature_options = [500, 1000, 2500, 5000, 7500]
    results = []

    print("--- Finding optimal max_features ---")
    print(f"{'Max Features':<15} | {'Validation F1-Score':<20}")
    print("-" * 40)

    for n_features in feature_options:
        # Create a new pipeline with the current max_features value
        # Note: We create a modified build_pipeline function for this
        pipeline = build_pipeline_with_params(
            prev_max_features=n_features,
            curr_max_features=n_features,
            sampling_strategy=sampling_strategy
        )

        # Train on the training set
        pipeline.fit(X_train, y_train)

        # Evaluate on the validation set
        y_val_pred = pipeline.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average='binary')

        print(f"{n_features:<15} | {val_f1:<20.4f}")
        results.append({'max_features': n_features, 'f1_score': val_f1})

    # Find the best result
    best_result = max(results, key=lambda x: x['f1_score'])
    print("-" * 40)
    print(f"Optimal max_features found: {best_result['max_features']} with F1-Score: {best_result['f1_score']:.4f}")

    return best_result['max_features']


def build_pipeline_with_params(prev_max_features=500, curr_max_features=500, sampling_strategy='none') -> ImbPipeline:
    """
    Builds a pipeline that includes preprocessing and an optional sampler.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('prev', TfidfVectorizer(ngram_range=(1, 2), max_features=prev_max_features), 'previous_clean'),
            ('curr', TfidfVectorizer(ngram_range=(1, 3), max_features=curr_max_features), 'current_clean')
        ],
        remainder='drop'
    )
    
    # Define the steps for the pipeline
    steps = [
        ('preprocessor', preprocessor)
    ]

    # Conditionally add a sampler step
    if sampling_strategy == 'oversample':
        steps.append(('sampler', RandomOverSampler(random_state=42)))
    elif sampling_strategy == 'undersample':
        steps.append(('sampler', RandomUnderSampler(random_state=42)))
        
    # Add the classifier WITHOUT class_weight='balanced'
    steps.append(('classifier', LogisticRegression(
        random_state=42, solver='liblinear', max_iter=1000
    )))
    
    # Use the imblearn Pipeline
    pipeline = ImbPipeline(steps)
    return pipeline


def main(args):
    """
    Main function to run the training workflow with proper train/val/test splits.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading training data from {args.training_file}...")
    try:
        df = pd.read_csv(args.training_file)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.training_file}")
        return

    # Fill empty sequences with an empty string
    df[['previous_clean', 'current_clean']] = df[['previous_clean', 'current_clean']].fillna('')
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    X = df[['previous_clean', 'current_clean']]
    y = df['label']
    
    # === PROPER TRAIN/VAL/TEST SPLIT ===
    print("\nSplitting data into train/val/test (60/20/20)...")
    
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: train/val from remaining 80% (60/20 of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # === BASELINE MODEL ===
    print("\n" + "="*50)
    print("TRAINING BASELINE MODEL (Keyword Lookup)")
    print("="*50)
    
    baseline = KeywordBaseline()
    baseline.fit(X_train, y_train)
    baseline_results = evaluate_model(baseline, X_test, y_test, "Keyword Baseline")
    
    # === ML MODEL ===
    print("\n" + "="*50)
    print("TRAINING ML MODEL (Logistic Regression + TF-IDF)")
    print("="*50)
    
    optimal_features = find_best_max_features(X_train, y_train, X_val, y_val, args.sampling)
    final_pipeline = build_pipeline_with_params(optimal_features, optimal_features, args.sampling)
    
    # Cross-validation on training set
    print("Running 5-fold cross-validation on training set...")
    cv_scoring = ['roc_auc', 'f1', 'precision', 'recall']
    cv_results = cross_validate(
        estimator=final_pipeline,
        X=X_train,
        y=y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=cv_scoring
    )
    
    # Train on full training set
    print("Training final model on full training set...")
    final_pipeline.fit(X_train, y_train)
    
    # Final evaluation on test set
    print("\nFinal test set performance:")
    ml_results = evaluate_model(final_pipeline, X_test, y_test, "ML Model (Test)")
    
    # === INFERENCE SPEED BENCHMARKING ===
    print("\n" + "="*50)
    print("INFERENCE SPEED BENCHMARKING")
    print("="*50)
    
    # Create sample data for benchmarking
    sample_data = pd.DataFrame({
        'previous_clean': ['So I was thinking about going to the store'] * 5,
        'current_clean': ['yeah', 'that sounds great', 'uh-huh', 'what time works?', 'mm-hmm']
    })
    
    print("\nBaseline speed:")
    baseline_speed = benchmark_inference_speed(baseline, sample_data)
    
    print("\nML Model speed:")
    ml_speed = benchmark_inference_speed(final_pipeline, sample_data)
    
    # === MODEL COMPARISON ===
    print(f"\n{'='*65}")
    print(f"{'MODEL COMPARISON SUMMARY':^65}")
    print(f"{'='*65}")
    print(f"{'Metric':<20} {'Baseline':<15} {'ML Model':<15} {'Improvement':<15}")
    print(f"{'-'*65}")
    print(f"{'ROC-AUC':<20} {baseline_results['roc_auc']:<15.4f} {ml_results['roc_auc']:<15.4f} {ml_results['roc_auc']-baseline_results['roc_auc']:+.4f}")
    print(f"{'F1-Score':<20} {baseline_results['f1_score']:<15.4f} {ml_results['f1_score']:<15.4f} {ml_results['f1_score']-baseline_results['f1_score']:+.4f}")
    print(f"{'Avg Precision':<20} {baseline_results['avg_precision']:<15.4f} {ml_results['avg_precision']:<15.4f} {ml_results['avg_precision']-baseline_results['avg_precision']:+.4f}")
    print(f"{'Speed (P95)':<20} {baseline_speed['p95_ms']:<15.2f} {ml_speed['p95_ms']:<15.2f} {ml_speed['p95_ms']-baseline_speed['p95_ms']:+.2f}")
    
    # === SAVE CONFUSION MATRICES ===
    plot_confusion_matrix(y_test, baseline_results['predictions'], 
                         "Keyword Baseline", 
                         os.path.join(args.output_dir, 'confusion_matrix_baseline.png'))
    
    plot_confusion_matrix(y_test, ml_results['predictions'], 
                         "ML Model", 
                         os.path.join(args.output_dir, 'confusion_matrix_ml.png'))
    
    # === GENERATE COMPREHENSIVE REPORT ===
    report_path = os.path.join(args.output_dir, 'training_report.txt')
    print(f"\nGenerating comprehensive training report at {report_path}...")
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BACKCHANNEL MODEL TRAINING REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Dataset: {os.path.basename(args.training_file)}\n")
        f.write(f"Sampling Strategy: {args.sampling}\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Train/Val/Test Split: {len(X_train)}/{len(X_val)}/{len(X_test)}\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("DATASET STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Original Class Distribution: {df['label'].value_counts().to_dict()}\n")
        if args.sampling != 'none':
            f.write(f"After {args.sampling}: {pd.Series(y_train).value_counts().to_dict()}\n")
        f.write("\n")
        
        f.write("CROSS-VALIDATION RESULTS (ML Model)\n")
        f.write("-" * 40 + "\n")
        for metric in cv_scoring:
            mean_score = np.mean(cv_results[f'test_{metric}'])
            std_score = np.std(cv_results[f'test_{metric}'])
            f.write(f"CV {metric.upper()}: {mean_score:.4f} (+/- {std_score:.4f})\n")
        f.write("\n")
        
        f.write("FINAL TEST SET PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"BASELINE RESULTS:\n")
        f.write(f"  ROC-AUC: {baseline_results['roc_auc']:.4f}\n")
        f.write(f"  F1-Score: {baseline_results['f1_score']:.4f}\n")
        f.write(f"  Avg Precision: {baseline_results['avg_precision']:.4f}\n")
        f.write(f"  Inference Speed (P95): {baseline_speed['p95_ms']:.2f}ms\n\n")
        
        f.write(f"ML MODEL RESULTS:\n")
        f.write(f"  ROC-AUC: {ml_results['roc_auc']:.4f}\n")
        f.write(f"  F1-Score: {ml_results['f1_score']:.4f}\n")
        f.write(f"  Avg Precision: {ml_results['avg_precision']:.4f}\n")
        f.write(f"  Inference Speed (P95): {ml_speed['p95_ms']:.2f}ms\n\n")
        
        f.write(f"IMPROVEMENTS:\n")
        f.write(f"  ROC-AUC: +{ml_results['roc_auc']-baseline_results['roc_auc']:.4f}\n")
        f.write(f"  F1-Score: +{ml_results['f1_score']-baseline_results['f1_score']:.4f}\n")
        f.write(f"  Avg Precision: +{ml_results['avg_precision']-baseline_results['avg_precision']:.4f}\n")
        f.write(f"  Speed Overhead: +{ml_speed['p95_ms']-baseline_speed['p95_ms']:.2f}ms\n\n")
        
        f.write("LATENCY REQUIREMENTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Target: <50ms\n")
        f.write(f"Baseline: {'✅ PASS' if baseline_speed['p95_ms'] < 50 else '❌ FAIL'}\n")
        f.write(f"ML Model: {'✅ PASS' if ml_speed['p95_ms'] < 50 else '❌ FAIL'}\n")

    # === SAVE MODELS ===
    model_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    baseline_filename = f"baseline_model_{model_timestamp}.joblib"
    baseline_path = os.path.join(args.output_dir, baseline_filename)
    joblib.dump(baseline, baseline_path)
    
    ml_filename = f"backchannel_model_{model_timestamp}.joblib"
    ml_path = os.path.join(args.output_dir, ml_filename)
    joblib.dump(final_pipeline, ml_path)
    
    print(f"\nModels saved:")
    print(f"  Baseline: {baseline_path}")
    print(f"  ML Model: {ml_path}")
    print(f"\nTraining completed successfully! 🎉")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a backchannel detection model with proper evaluation.")
    parser.add_argument('--training-file', type=str, required=True, 
                       help="Path to the training CSV dataset with columns: previous_clean, current_clean, label")
    parser.add_argument('--output-dir', type=str, default='./output', 
                       help="Directory to save the trained model and report.")
    parser.add_argument('--sampling', type=str, choices=['none', 'oversample', 'undersample'], 
                       default='none', help="Sampling strategy for class imbalance")
    
    args = parser.parse_args()
    main(args)