import torch
import numpy as np
from tqdm import tqdm
import json
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def calculate_ece(preds, labels, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(preds, axis=1)
    accuracies = (np.argmax(preds, axis=1) == labels).astype(int)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def calculate_ppl(model, tokenizer, dataset_name, split, device, max_samples=1000):
    """Calculate perplexity on a dataset."""
    try:
        dataset = load_dataset(dataset_name, 'en', split=split, streaming=True)
    except Exception:
        # Fallback for datasets like wikitext
        dataset = load_dataset(dataset_name, 'wikitext-2-raw-v1', split=split, streaming=True)
        
    dataset = dataset.take(max_samples)
    
    model.eval()
    nlls = []
    stride = 512
    max_length = model.config.max_position_embeddings

    for row in tqdm(dataset, desc=f"Evaluating PPL on {dataset_name}"):
        text = row['text']
        if not text:
            continue
        encodings = tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def run_evaluation(model, config, device):
    """Runs all specified evaluations and prints results."""
    model.eval()
    results = {}
    eval_config = config.get('evaluation', {})
    tokenizer_name = config['model'].get('tokenizer_name') or config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ECE on WikiText-2
    if 'wikitext2_ece' in eval_config.get('tasks', []):
        print("\n--- Evaluating ECE on WikiText-2 ---")
        try:
            wikitext_val = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
            wikitext_encodings = tokenizer('\n\n'.join(wikitext_val['text']), return_tensors='pt')
            
            all_preds = []
            all_labels = []
            max_length = 512
            stride = 256
            seq_len = wikitext_encodings.input_ids.size(1)

            for i in tqdm(range(0, seq_len - max_length + 1, stride)):
                input_ids = wikitext_encodings.input_ids[:, i:i+max_length].to(device)
                labels = wikitext_encodings.input_ids[:, i+1:i+max_length+1].reshape(-1).cpu().numpy()
                
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                
                preds = torch.softmax(logits, dim=-1)[0, :-1, :].reshape(-1, logits.size(-1)).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)
            
            ece = calculate_ece(np.concatenate(all_preds), np.concatenate(all_labels))
            results['wikitext2_ece'] = ece
            print(f"WikiText-2 ECE: {ece:.4f}")
        except Exception as e:
            print(f"Could not evaluate ECE on WikiText-2: {e}")

    # Perplexity on C4
    if 'c4_ppl' in eval_config.get('tasks', []):
        print("\n--- Evaluating PPL on C4 ---")
        try:
            c4_ppl = calculate_ppl(model, tokenizer, 'allenai/c4', 'validation', device)
            results['c4_ppl'] = c4_ppl
            print(f"C4 Perplexity: {c4_ppl:.4f}")
        except Exception as e:
            print(f"Could not evaluate PPL on C4: {e}")
    
    print("\n--- Evaluation Summary ---")
    print(json.dumps(results, indent=2))
    return results
