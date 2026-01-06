#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: classification_head.py
# Description: Classification head for end-to-end VQA (direct answer prediction)
# 
# This module replaces the seq2seq decoder for direct answer classification,
# bypassing the symbolic executor entirely.

import torch
import torch.nn as nn

__all__ = ['ClassificationHead']


class ClassificationHead(nn.Module):
    """
    MLP Classification Head for direct VQA answer prediction.
    
    Takes the fused graph embedding from GraphMatcher and predicts
    the answer directly (e.g., "blue", "yes", "3") without generating
    a program sequence.
    
    Architecture:
        g_embd (96D) -> Linear -> ReLU -> Dropout -> Linear -> num_classes
    
    Args:
        input_dim (int): Dimension of input embedding (default: 96 from GraphMatcher)
        hidden_dim (int): Dimension of hidden layer (default: 256)
        num_classes (int): Number of answer classes (CLEVR has ~28-32)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_dim=96, hidden_dim=256, num_classes=28, dropout=0.1):
        super(ClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # MLP: input -> hidden -> output
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, g_embd):
        """
        Forward pass for classification.
        
        Args:
            g_embd (Tensor): Graph embedding from GraphMatcher, shape [batch_size, input_dim]
        
        Returns:
            logits (Tensor): Raw logits for each class, shape [batch_size, num_classes]
        """
        # Ensure g_embd is 2D (batch_size, input_dim)
        if len(g_embd.shape) == 3:
            # If 3D (batch, nodes, dim), take mean over nodes
            g_embd = torch.mean(g_embd, dim=1)
        
        logits = self.classifier(g_embd)
        return logits
    
    def predict(self, g_embd):
        """
        Get predicted class indices.
        
        Args:
            g_embd (Tensor): Graph embedding from GraphMatcher
        
        Returns:
            predictions (Tensor): Predicted class indices, shape [batch_size]
        """
        logits = self.forward(g_embd)
        return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, g_embd):
        """
        Get class probabilities.
        
        Args:
            g_embd (Tensor): Graph embedding from GraphMatcher
        
        Returns:
            probs (Tensor): Class probabilities, shape [batch_size, num_classes]
        """
        logits = self.forward(g_embd)
        return torch.softmax(logits, dim=-1)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  input_dim={self.input_dim},\n'
                f'  hidden_dim={self.hidden_dim},\n'
                f'  num_classes={self.num_classes}\n'
                f')')
