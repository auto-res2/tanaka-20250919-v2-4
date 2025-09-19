import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, get_scheduler
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import copy
import os
from tqdm import tqdm

class AIBLLoss(nn.Module):
    """Adaptive Information-Balanced Loss (AIBL)."""
    def __init__(self, gamma=3.0, lambda_r=0.01, reduction='mean', layer_idx=15):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive for log(G) to be defined.")
        self.gamma, self.lambda_r = gamma, lambda_r
        self.reduction, self.layer_idx = reduction, layer_idx

    def forward(self, logits, target, base_logits, reps, base_reps):
        # logits/base_logits: [B,T,V]; reps/base_reps: [B,T,D]
        # Ensure inputs are in float32 for stable calculations
        p = logits.float().softmax(-1)
        p0 = base_logits.float().softmax(-1)

        # spherical score
        one_hot = F.one_hot(target, logits.size(-1)).float()
        sph = 1 - (p * one_hot).sum(-1) / (p.norm(dim=-1) + 1e-9)

        # KL divergence
        kl = (p * (p.log() - p0.log())).sum(-1)

        # Fisher-ratio gate (trace of diag(p)-pp^T times inverse)
        # The trace of the Fisher Information Matrix for a categorical distribution is sum(p*(1-p))
        fisher = p * (1 - p)
        fisher0 = p0 * (1 - p0)
        G = (fisher / (fisher0 + 1e-9)).sum(-1)
        
        # Clamp G to avoid log(0) or log(<0)
        G = torch.clamp(G, min=1e-9)
        w = torch.sigmoid(self.gamma * G.log())

        # representation anchor
        rep_loss = F.mse_loss(reps, base_reps, reduction='none').mean(-1)

        # Combine losses
        loss = w * sph + (1 - w) * (kl + self.lambda_r * rep_loss)

        if self.reduction == 'mean':
            # Mask out padding tokens (assuming target is -100 for padding)
            mask = (target != -100).float()
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

def setup_models_and_optimizer(config, device):
    """Loads base model, applies PEFT, creates a frozen copy, and sets up optimizer."""
    model_name = config['model']['name']
    torch_dtype = torch.bfloat16 if config['model']['torch_dtype'] == 'bfloat16' else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_auth_token=os.getenv("HF_TOKEN"),
        trust_remote_code=True,
    )
    
    # Example used NF4 quantization, we'll follow that
    model = prepare_model_for_kbit_training(model)
    
    peft_config = config['peft']
    lora_cfg = LoraConfig(
        r=peft_config['r'],
        lora_alpha=peft_config['lora_alpha'],
        target_modules=peft_config['target_modules'],
        lora_dropout=peft_config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    base_model = copy.deepcopy(model)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    optim_config = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=optim_config['lr'], 
        betas=(optim_config['adam_beta1'], optim_config['adam_beta2']),
        eps=optim_config['adam_eps']
    )
    
    return model, base_model, optimizer

def run_training_loop(config, model, base_model, train_loader, val_loader, optimizer, device):
    """Runs the main training and validation loop."""
    train_config = config['training']
    num_training_steps = len(train_loader) * train_config['epochs']
    
    scheduler = get_scheduler(
        name=train_config['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=train_config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    loss_fn_config = config['aibl_loss']
    loss_fn = AIBLLoss(
        gamma=loss_fn_config['gamma'],
        lambda_r=loss_fn_config['lambda_r'],
        layer_idx=loss_fn_config['layer_idx']
    ).to(device)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(train_config['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.get('labels')

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True)
                with torch.no_grad():
                    base_outputs = base_model(**inputs, output_hidden_states=True)
                
                loss = loss_fn(
                    logits=outputs.logits,
                    target=labels,
                    base_logits=base_outputs.logits,
                    reps=outputs.hidden_states[loss_fn.layer_idx],
                    base_reps=base_outputs.hidden_states[loss_fn.layer_idx]
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

    print("Training complete.")
    return model
