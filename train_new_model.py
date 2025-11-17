"""
Hausa-English Translation Model Training Script
Optimized for M1 MacBook Air (8GB)
Research-Grade with Terminal Logging
"""

import torch
from transformers import (
    MarianMTModel, 
    MarianTokenizer, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
import sacrebleu
import re
import unicodedata
import numpy as np
import os
import sys
import json
from datetime import datetime
from collections import defaultdict

# CONFIGURATION

CONFIG = {
    # File paths
    "data_file": "Eng_Hausa.csv",
    "output_dir": "./results",
    "model_dir": "./final_model",
    "metrics_file": "./training_metrics.json",
    "summary_file": "./training_summary.txt",
    
    # Model
    "model_name": "Helsinki-NLP/opus-mt-ha-en",
    "max_length": 40,
    
    # Training
    "num_epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 3e-5,
    "warmup_steps": 300,
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 25,
    
    # Data filtering
    "min_text_length": 5,
    "max_text_length": 80,
}

# LOGGING SETUP

class TerminalLogger:
    """Terminal-only logging for research purposes"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics_history = defaultdict(list)
        self.epoch_stats = []
        
        self.print_header("HAUSA-ENGLISH TRANSLATION MODEL TRAINING")
        self.print_header("Research-Grade Training - Terminal Output")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def print_header(self, text):
        """Print formatted section header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)
    
    def print_subheader(self, text):
        """Print formatted subsection"""
        print("\n" + "-"*70)
        print(f"  {text}")
        print("-"*70)
    
    def log(self, message):
        """Print message to terminal"""
        print(message)
    
    def log_config(self, config):
        """Log configuration parameters"""
        self.print_header("CONFIGURATION PARAMETERS")
        for key, value in config.items():
            print(f"  {key:30s}: {value}")
    
    def log_dataset_stats(self, stats):
        """Log dataset statistics"""
        self.print_header("DATASET STATISTICS")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.2f}")
            else:
                print(f"  {key:30s}: {value}")
    
    def log_model_info(self, model, tokenizer):
        """Log model information"""
        self.print_header("MODEL INFORMATION")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Model Name:               {CONFIG['model_name']}")
        print(f"  Total Parameters:         {total_params:,}")
        print(f"  Trainable Parameters:     {trainable_params:,}")
        print(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
        print(f"  Model dtype:              {model.dtype}")
        print(f"  Device:                   {model.device}")
        print(f"  Vocab Size:               {len(tokenizer)}")
    
    def log_training_step(self, step, total_steps, metrics):
        """Log training step metrics"""
        self.metrics_history['step'].append(step)
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        progress = (step / total_steps) * 100
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {step}/{total_steps} ({progress:.1f}%) | {metrics_str}")
    
    def log_epoch_summary(self, epoch, metrics):
        """Log epoch summary"""
        self.print_subheader(f"EPOCH {epoch} SUMMARY")
        
        epoch_data = {"epoch": epoch, "metrics": metrics}
        self.epoch_stats.append(epoch_data)
        
        for key, value in metrics.items():
            print(f"  {key:25s}: {value:.4f}")
    
    def save_metrics(self):
        """Save all metrics to JSON"""
        data = {
            "config": CONFIG,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "metrics_history": dict(self.metrics_history),
            "epoch_stats": self.epoch_stats
        }
        
        with open(CONFIG["metrics_file"], 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Metrics saved to: {CONFIG['metrics_file']}")
    
    def generate_summary(self, final_metrics):
        """Generate training summary report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = []
        summary.append("="*70)
        summary.append("TRAINING SUMMARY REPORT")
        summary.append("="*70)
        summary.append("")
        
        # Time info
        summary.append("TIMING INFORMATION:")
        summary.append(f"  Start Time:     {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"  End Time:       {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"  Total Duration: {duration}")
        summary.append(f"  Hours:          {duration.total_seconds() / 3600:.2f}h")
        summary.append("")
        
        # Configuration
        summary.append("CONFIGURATION:")
        for key, value in CONFIG.items():
            summary.append(f"  {key:30s}: {value}")
        summary.append("")
        
        # Final metrics
        summary.append("FINAL METRICS:")
        for key, value in final_metrics.items():
            summary.append(f"  {key:25s}: {value:.4f}")
        summary.append("")
        
        # Epoch progression
        if self.epoch_stats:
            summary.append("EPOCH PROGRESSION:")
            summary.append(f"  {'Epoch':<8} {'Loss':<12} {'Eval Loss':<12} {'BLEU':<12}")
            summary.append("  " + "-"*50)
            for epoch_data in self.epoch_stats:
                epoch = epoch_data['epoch']
                metrics = epoch_data['metrics']
                loss = metrics.get('train_loss', 0)
                eval_loss = metrics.get('eval_loss', 0)
                bleu = metrics.get('eval_bleu', 0)
                summary.append(f"  {epoch:<8} {loss:<12.4f} {eval_loss:<12.4f} {bleu:<12.4f}")
            summary.append("")
        
        # Best metrics
        if self.epoch_stats:
            best_bleu_epoch = max(self.epoch_stats, key=lambda x: x['metrics'].get('eval_bleu', 0))
            best_loss_epoch = min(self.epoch_stats, key=lambda x: x['metrics'].get('train_loss', float('inf')))
            
            summary.append("BEST PERFORMANCE:")
            summary.append(f"  Best BLEU Score:  {best_bleu_epoch['metrics'].get('eval_bleu', 0):.4f} (Epoch {best_bleu_epoch['epoch']})")
            summary.append(f"  Lowest Loss:      {best_loss_epoch['metrics'].get('train_loss', 0):.4f} (Epoch {best_loss_epoch['epoch']})")
            summary.append("")
        
        summary.append("="*70)
        
        summary_text = "\n".join(summary)
        
        # Save to file
        with open(CONFIG["summary_file"], 'w') as f:
            f.write(summary_text)
        
        # Print to terminal
        print("\n" + summary_text)
        
        return summary_text

# Initialize logger
logger = TerminalLogger()

# SETUP

def setup_device():
    """Detect and configure device (MPS/CUDA/CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        print("✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
    
    return device

def print_memory_usage():
    """Print current memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB)")
    except:
        pass

# DATA CLEANING

def clean_text(text):
    """Clean text for translation"""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", text)
    text = " ".join(text.split())
    text = text.strip()
    return text

def clean_dataset(df, hausa_col='HAUSA', english_col='ENGLISH'):
    """Clean and filter translation dataset"""
    logger.print_header("DATA CLEANING PROCESS")
    print(f"Initial dataset size: {len(df):,} rows\n")
    
    df_clean = df.copy()
    initial_size = len(df)
    
    # Track removed rows
    removal_stats = {}
    
    # Apply text cleaning
    print("Applying text normalization...")
    df_clean[hausa_col] = df_clean[hausa_col].apply(clean_text)
    df_clean[english_col] = df_clean[english_col].apply(clean_text)
    
    # Remove duplicates
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[hausa_col, english_col])
    removed = before - len(df_clean)
    removal_stats['duplicates'] = removed
    print(f"  ✓ Removed {removed:,} duplicates ({removed/initial_size*100:.1f}%)")
    
    # Remove empty strings
    before = len(df_clean)
    df_clean = df_clean[
        (df_clean[hausa_col].str.len() > 0) &
        (df_clean[english_col].str.len() > 0)
    ]
    removed = before - len(df_clean)
    removal_stats['empty'] = removed
    print(f"  ✓ Removed {removed:,} empty rows ({removed/initial_size*100:.1f}%)")
    
    # Remove very short texts
    before = len(df_clean)
    df_clean = df_clean[
        (df_clean[hausa_col].str.len() >= CONFIG["min_text_length"]) &
        (df_clean[english_col].str.len() >= CONFIG["min_text_length"])
    ]
    removed = before - len(df_clean)
    removal_stats['too_short'] = removed
    print(f"  ✓ Removed {removed:,} rows - too short ({removed/initial_size*100:.1f}%)")
    
    # Remove very long texts
    before = len(df_clean)
    df_clean = df_clean[
        (df_clean[hausa_col].str.len() <= CONFIG["max_text_length"]) &
        (df_clean[english_col].str.len() <= CONFIG["max_text_length"])
    ]
    removed = before - len(df_clean)
    removal_stats['too_long'] = removed
    print(f"  ✓ Removed {removed:,} rows - too long ({removed/initial_size*100:.1f}%)")
    
    # Remove suspicious length ratios
    df_clean['len_ratio'] = df_clean[hausa_col].str.len() / df_clean[english_col].str.len()
    before = len(df_clean)
    df_clean = df_clean[
        (df_clean['len_ratio'] >= 0.3) &
        (df_clean['len_ratio'] <= 3.0)
    ]
    df_clean = df_clean.drop('len_ratio', axis=1)
    removed = before - len(df_clean)
    removal_stats['suspicious_ratio'] = removed
    print(f"  ✓ Removed {removed:,} rows - suspicious ratio ({removed/initial_size*100:.1f}%)")
    
    # Rename and reset
    df_clean = df_clean.rename(columns={hausa_col: 'Hausa', english_col: 'English'})
    df_clean = df_clean.reset_index(drop=True)
    
    final_size = len(df_clean)
    retention_rate = (final_size / initial_size) * 100
    
    print(f"\n✓ Final dataset: {final_size:,} rows ({retention_rate:.1f}% retained)")
    print(f"✓ Total removed: {initial_size - final_size:,} rows")
    
    # Calculate statistics
    stats = {
        "initial_rows": initial_size,
        "final_rows": final_size,
        "removed_rows": initial_size - final_size,
        "retention_rate": f"{retention_rate:.2f}%",
        "hausa_min_length": int(df_clean['Hausa'].str.len().min()),
        "hausa_max_length": int(df_clean['Hausa'].str.len().max()),
        "hausa_mean_length": float(df_clean['Hausa'].str.len().mean()),
        "hausa_median_length": float(df_clean['Hausa'].str.len().median()),
        "english_min_length": int(df_clean['English'].str.len().min()),
        "english_max_length": int(df_clean['English'].str.len().max()),
        "english_mean_length": float(df_clean['English'].str.len().mean()),
        "english_median_length": float(df_clean['English'].str.len().median()),
    }
    
    # Add removal stats
    for key, value in removal_stats.items():
        stats[f"removed_{key}"] = value
    
    logger.log_dataset_stats(stats)
    
    # Show samples
    print("\nSample cleaned data:")
    for i in range(min(5, len(df_clean))):
        print(f"  [{i+1}] HA: {df_clean.iloc[i]['Hausa'][:60]}...")
        print(f"      EN: {df_clean.iloc[i]['English'][:60]}...")
    
    return df_clean, stats

# MODEL & TOKENIZATION

def load_model_and_tokenizer(device):
    """Load pretrained model and tokenizer"""
    logger.print_header("LOADING MODEL")
    
    model_name = CONFIG["model_name"]
    print(f"Model: {model_name}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Load model
    print("Loading model (this may take a minute if downloading)...")
    model = MarianMTModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    
    # Move to device
    model = model.to(device)
    
    print("✓ Model loaded successfully")
    logger.log_model_info(model, tokenizer)
    
    return model, tokenizer

def preprocess_function(examples, tokenizer):
    """Tokenize inputs and targets"""
    max_length = CONFIG["max_length"]
    
    model_inputs = tokenizer(
        examples['Hausa'],
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    labels = tokenizer(
        text_target=examples['English'],
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# EVALUATION

def compute_metrics(eval_pred, tokenizer):
    """Compute BLEU score"""
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    
    return {"bleu": bleu.score}

# TRAINING

class MetricsCallback(EarlyStoppingCallback):
    """Custom callback to log metrics during training"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging"""
        if logs:
            metrics = {}
            if 'loss' in logs:
                metrics['loss'] = logs['loss']
            if 'learning_rate' in logs:
                metrics['lr'] = logs['learning_rate']
            if 'epoch' in logs:
                self.current_epoch = logs['epoch']
                metrics['epoch'] = logs['epoch']
            
            if metrics and state.global_step % CONFIG['logging_steps'] == 0:
                total_steps = state.max_steps
                logger.log_training_step(state.global_step, total_steps, metrics)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics:
            logger.print_subheader(f"EVALUATION at Step {state.global_step}")
            for key, value in metrics.items():
                print(f"  {key:20s}: {value:.4f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at end of each epoch"""
        epoch = int(self.current_epoch)
        if hasattr(state, 'log_history'):
            recent_logs = [l for l in state.log_history if l.get('epoch', 0) >= epoch - 0.1]
            if recent_logs:
                epoch_metrics = {}
                for log in recent_logs:
                    for key, value in log.items():
                        if isinstance(value, (int, float)) and key != 'epoch':
                            if key not in epoch_metrics:
                                epoch_metrics[key] = []
                            epoch_metrics[key].append(value)
                
                avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
                logger.log_epoch_summary(epoch, avg_metrics)

def create_trainer(model, tokenizer, train_dataset, eval_dataset, device):
    """Create and configure Seq2SeqTrainer"""
    logger.print_header("CONFIGURING TRAINER")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG["output_dir"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["num_epochs"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        lr_scheduler_type="linear",
        optim="adamw_torch",
        max_grad_norm=1.0,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        fp16=False,
        predict_with_generate=True,
        generation_max_length=CONFIG["max_length"],
        generation_num_beams=1,
        logging_dir="./logs",
        logging_steps=CONFIG["logging_steps"],
        logging_first_step=True,
        report_to="none",
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        seed=42,
        use_mps_device=True if device.type == "mps" else False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=[MetricsCallback(early_stopping_patience=5)]
    )
    
    # Calculate training info
    total_steps = len(train_dataset) // (CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']) * CONFIG['num_epochs']
    steps_per_epoch = total_steps // CONFIG['num_epochs']
    
    print(f"Training Configuration:")
    print(f"  Epochs:                   {CONFIG['num_epochs']}")
    print(f"  Batch size:               {CONFIG['batch_size']}")
    print(f"  Gradient accumulation:    {CONFIG['gradient_accumulation_steps']}")
    print(f"  Effective batch size:     {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate:            {CONFIG['learning_rate']}")
    print(f"  Warmup steps:             {CONFIG['warmup_steps']}")
    print(f"  Total steps:              {total_steps:,}")
    print(f"  Steps per epoch:          {steps_per_epoch:,}")
    print(f"  Training samples:         {len(train_dataset):,}")
    print(f"  Eval samples:             {len(eval_dataset):,}")
    
    # Estimate time
    estimated_seconds = total_steps * 4.5
    estimated_hours = estimated_seconds / 3600
    print(f"\nEstimated training time: {estimated_hours:.1f} hours ({estimated_seconds/60:.0f} minutes)")
    
    return trainer

# TESTING

def test_translation(model, tokenizer, device):
    """Test the trained model"""
    logger.print_header("TESTING TRANSLATIONS")
    
    test_texts = [
        "Ina kwana?",
        "Yaya kake?",
        "Na gode sosai",
        "Ina son ka",
        "Barka da zuwa",
        "Ina son ruwa",
        "Yaushe za ku zo?",
        "Wannan littafi ne mai kyau"
    ]
    
    model.eval()
    
    for i, text in enumerate(test_texts, 1):
        with torch.no_grad():
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                max_length=CONFIG["max_length"], 
                truncation=True
            ).to(device)
            
            outputs = model.generate(
                **inputs, 
                max_length=CONFIG["max_length"], 
                num_beams=4, 
                early_stopping=True
            )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"{i}. Hausa:   {text}")
        print(f"   English: {translation}\n")

# MAIN

def main():
    """Main training pipeline"""
    
    # Log configuration
    logger.log_config(CONFIG)
    
    # Setup
    print("")
    device = setup_device()
    print_memory_usage()
    
    # Load data
    logger.print_header("LOADING DATA")
    
    if not os.path.exists(CONFIG["data_file"]):
        print(f"Error: {CONFIG['data_file']} not found!")
        print(f"Current directory: {os.getcwd()}")
        print("\nAvailable CSV files:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  - {f}")
        sys.exit(1)
    
    print(f"Reading {CONFIG['data_file']}...")
    df = pd.read_csv(CONFIG["data_file"])
    print(f"✓ Loaded {len(df):,} rows")
    
    # Clean data
    df_cleaned, dataset_stats = clean_dataset(df)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(device)
    
    # Prepare dataset
    logger.print_header("PREPARING DATASET")
    print("Creating HuggingFace Dataset...")
    dataset = Dataset.from_pandas(df_cleaned)
    
    print("Tokenizing (this may take a few minutes)...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print("Splitting train/eval...")
    split = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    print(f"✓ Train: {len(train_dataset):,} samples")
    print(f"✓ Eval:  {len(eval_dataset):,} samples")
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, device)
    
    # Train
    logger.print_header("STARTING TRAINING")
    print("Training for 10 epochs - Expected time: ~2 hours on M1 MacBook Air")
    print("Make sure your Mac is plugged in and won't sleep")
    print("\nTraining progress:\n")
    
    try:
        trainer.train()
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current model state...")
        trainer.save_model(CONFIG["model_dir"])
        tokenizer.save_pretrained(CONFIG["model_dir"])
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save model
    logger.print_header("SAVING MODEL")
    print(f"Saving to {CONFIG['model_dir']}...")
    trainer.save_model(CONFIG["model_dir"])
    tokenizer.save_pretrained(CONFIG["model_dir"])
    print("✓ Model saved successfully!")
    
    # Save metrics
    logger.save_metrics()
    
    # Get final metrics
    final_metrics = {}
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        final_metrics = {k: v for k, v in last_log.items() if isinstance(v, (int, float)) and k != 'epoch'}
    
    # Generate summary
    logger.generate_summary(final_metrics)
    
    # Test translations
    test_translation(model, tokenizer, device)
    
    # Final stats
    end_time = datetime.now()
    duration = end_time - logger.start_time
    
    logger.print_header("FINAL SUMMARY")
    print(f"Start time:        {logger.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:          {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration:    {duration}")
    print(f"Device:            {device}")
    print(f"Final model:       {CONFIG['model_dir']}")
    print(f"Metrics saved to:  {CONFIG['metrics_file']}")
    print(f"Summary saved to:  {CONFIG['summary_file']}")
    print_memory_usage()
    print("\n✓ All done! 🎉\n")

if __name__ == "__main__":
    main()