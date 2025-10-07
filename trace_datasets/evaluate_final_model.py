#!/usr/bin/env python3
"""
Evaluate the final SFT model on all TRACE tasks using TRACE evaluation infrastructure.
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add TRACE to path for evaluation functions
sys.path.append('/workspace/TRACE')
from evaluations import eval_CStance, eval_FOMC, eval_MeetingBank, eval_ScienceQA
from evaluations import eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten

def load_test_data(test_file, max_samples=None):
    """Load test data from our converted JSONL files"""
    data = []
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line.strip())
            # Extract prompt and ground truth from mini_trainer format
            user_msg = item['messages'][0]['content']
            ground_truth = item['messages'][1]['content']
            data.append({
                'prompt': user_msg,
                'ground_truth': ground_truth
            })
    return data

def generate_predictions(model, tokenizer, prompts, max_length=2048):
    """Generate predictions for a list of prompts using proper chat template"""
    device = next(model.parameters()).device
    predictions = []

    for prompt in tqdm(prompts, desc="Generating predictions"):
        # Format prompt as conversation using chat template
        conversation = [{"role": "user", "content": prompt}]

        # Apply chat template to get properly formatted prompt
        formatted_prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the formatted prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode only the generated part
        prompt_length = inputs['input_ids'].shape[1]
        generated = outputs[0][prompt_length:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

        predictions.append(prediction)

    return predictions

def evaluate_dataset(dataset_name, test_file, model, tokenizer):
    """Evaluate model on a single dataset"""
    print(f"\\n{'='*50}")
    print(f"Evaluating {dataset_name}")
    print(f"{'='*50}")

    # Load test data
    test_data = load_test_data(test_file)
    prompts = [item['prompt'] for item in test_data]
    ground_truths = [item['ground_truth'] for item in test_data]

    print(f"Loaded {len(test_data)} test samples")

    # Generate predictions using efficient batching
    predictions = generate_predictions(model, tokenizer, prompts)

    # Evaluate using TRACE functions
    if dataset_name == "C-STANCE":
        results = eval_CStance.eval(predictions, ground_truths)
    elif dataset_name == "FOMC":
        results = eval_FOMC.eval(predictions, ground_truths)
    elif dataset_name == "MeetingBank":
        results = eval_MeetingBank.eval(predictions, ground_truths)
    elif dataset_name == "ScienceQA":
        results = eval_ScienceQA.eval(predictions, ground_truths)
    elif dataset_name == "NumGLUE-cm":
        results = eval_NumGLUE_cm.eval(predictions, ground_truths)
    elif dataset_name == "NumGLUE-ds":
        results = eval_NumGLUE_ds.eval(predictions, ground_truths)
    elif dataset_name == "20Minuten":
        # 20Minuten needs source sequences too (original text for SARI)
        source_sequences = prompts  # Use prompts as source for simplification
        results = eval_20Minuten.eval(source_sequences, predictions, ground_truths)
    else:
        results = {"error": "Unknown dataset"}

    print(f"Results for {dataset_name}: {results}")
    return results

def main():
    """Main evaluation function"""
    print("🎯 TRACE Benchmark Evaluation")
    print("="*60)

    # Model path (temporarily using base model for testing)
    # model_path = "meta-llama/Llama-2-7b-chat-hf"
    model_path = "/workspace/mini_trainer/trace_outputs/OSFT_fixed/hf_format/samples_2500.0"
    print(f"📂 Model: {model_path}")

    # Check if model exists (skip check for HF models)
    if model_path.startswith("/") and not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run the SFT training first!")
        return

    # Load model and tokenizer
    print("🔄 Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test datasets to evaluate (for testing, using only NumGLUE-cm)
    datasets = [
        ("C-STANCE", "/workspace/mini_trainer/trace_datasets/converted/C-STANCE_test.jsonl"),
        # ("FOMC", "/workspace/mini_trainer/trace_datasets/converted/FOMC_test.jsonl"),
        # ("MeetingBank", "/workspace/mini_trainer/trace_datasets/converted/MeetingBank_test.jsonl"),
        ("ScienceQA", "/workspace/mini_trainer/trace_datasets/converted/ScienceQA_test.jsonl"),
        ("NumGLUE-cm", "/workspace/mini_trainer/trace_datasets/converted/NumGLUE-cm_test.jsonl"),
        ("NumGLUE-ds", "/workspace/mini_trainer/trace_datasets/converted/NumGLUE-ds_test.jsonl"),
        # ("20Minuten", "/workspace/mini_trainer/trace_datasets/converted/20Minuten_test.jsonl"),
    ]

    # Evaluate each dataset
    all_results = {}
    accuracies = []

    for dataset_name, test_file in datasets:
        if not os.path.exists(test_file):
            print(f"⚠️  Test file not found: {test_file}")
            continue

        results = evaluate_dataset(dataset_name, test_file, model, tokenizer)
        all_results[dataset_name] = results

        # Extract primary metric (accuracy for most, rouge-L for others)
        if 'accuracy' in results:
            accuracies.append(results['accuracy'])
        elif 'sari' in results:
            accuracies.append(results['sari']/100)
        elif 'rouge-L' in results:
            accuracies.append(results['rouge-L'])

    # Compute average
    if accuracies:
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"\\n🎯 FINAL RESULTS:")
        print(f"{'='*60}")
        print(f"📊 Average Performance: {avg_accuracy:.4f}")
        print(f"📝 Individual Results:")
        for dataset_name, results in all_results.items():
            main_metric = results.get('accuracy', results.get('sari', results.get('rouge-L', 'N/A')))
            print(f"  {dataset_name}: {main_metric}")
        print(f"{'='*60}")

    # Save detailed results
    output_file = "sft_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model_path': model_path,
            'average_performance': avg_accuracy if accuracies else None,
            'individual_results': all_results
        }, f, indent=2)

    print(f"💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()