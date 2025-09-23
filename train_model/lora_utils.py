import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

class LoRALayer(nn.Module):
    """LoRA层的基类"""
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

class LoRALinear(nn.Module):
    """LoRA线性层"""
    def __init__(self, linear_layer: nn.Linear, r: int = 16, lora_alpha: int = 32, 
                 lora_dropout: float = 0.1, fan_in_fan_out: bool = False):
        super().__init__()
        self.linear_layer = linear_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.fan_in_fan_out = fan_in_fan_out
        
        # 冻结原始参数
        for param in self.linear_layer.parameters():
            param.requires_grad = False
            
        # LoRA参数
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = lora_alpha / r
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # 原始线性层输出
        original_output = self.linear_layer(x)
        
        # LoRA输出
        lora_output = F.dropout(x, p=self.lora_dropout, training=self.training)
        lora_output = lora_output @ self.lora_A.T
        lora_output = lora_output @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output

class LoRAConfig:
    """LoRA配置类"""
    def __init__(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1,
                 target_modules: Optional[list] = None, bias: str = "none"):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.bias = bias

def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """将LoRA应用到模型"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是目标模块
            if any(target in name for target in config.target_modules):
                # 替换为LoRA层
                lora_layer = LoRALinear(
                    module, 
                    r=config.r, 
                    lora_alpha=config.lora_alpha, 
                    lora_dropout=config.lora_dropout
                )
                # 更新模型中的模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, child_name, lora_layer)
                else:
                    setattr(model, child_name, lora_layer)
    
    return model

def get_lora_parameters(model: nn.Module) -> list:
    """获取LoRA参数"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
    return lora_params

def save_lora_weights(model: nn.Module, save_path: str):
    """保存LoRA权重"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict[name] = param.data
    torch.save(lora_state_dict, save_path)

def load_lora_weights(model: nn.Module, load_path: str):
    """加载LoRA权重"""
    lora_state_dict = torch.load(load_path)
    model_state_dict = model.state_dict()
    
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param
    
    model.load_state_dict(model_state_dict) 