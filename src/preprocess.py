from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os

def get_dataloaders(config):
    """Loads and preprocesses a single dataset task, returning DataLoaders."""
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.get('tokenizer_name') or model_config['name'],
            use_auth_token=os.getenv("HF_TOKEN"),
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        # Generic pre-processing for summarization style tasks
        inputs = [f"Summarize: {article}" for article in examples['article']]
        targets = [summary for summary in examples['highlights']]
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(inputs, max_length=data_config['max_input_length'], truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=data_config['max_target_length'], truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(f"Loading dataset: {data_config['dataset_name']}")
    raw_datasets = load_dataset(data_config['dataset_name'], data_config['dataset_config'])
    
    # Select subsets for faster processing if specified
    if 'train_samples' in data_config and data_config['train_samples'] > 0:
        raw_datasets['train'] = raw_datasets['train'].select(range(data_config['train_samples']))
    if 'val_samples' in data_config and data_config['val_samples'] > 0:
        raw_datasets['validation'] = raw_datasets['validation'].select(range(data_config['val_samples']))
    if 'test_samples' in data_config and data_config['test_samples'] > 0:
        raw_datasets['test'] = raw_datasets['test'].select(range(data_config['test_samples']))

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=['article', 'highlights', 'id'])
    tokenized_datasets.set_format(type='torch')

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=training_config['batch_size'])
    val_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=training_config['batch_size'])
    
    print("Dataloaders created successfully.")
    return train_dataloader, val_dataloader
