#! /usr/bin/env python
# -*- coding: utf-8 -*-
# File: classifier_model.py
# Description: Classifier model for end-to-end VQA (direct answer prediction)
#
# This model bypasses the seq2seq decoder and executor entirely,
# predicting the answer directly from graph embeddings.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Data, Batch

from . import get_vocab
from . import create_classifier_net

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

__all__ = ['ClassifierModel']


class ClassifierModel:
    """
    Model interface for end-to-end VQA classification.
    
    This model uses GNN + GraphMatcher + ClassificationHead to predict
    answers directly, without program generation or symbolic execution.
    
    Architecture:
        Graphs (Gs, Gt) -> GNN -> GraphMatcher -> g_embd -> MLP -> answer
    """

    def __init__(self, opt, num_answer_classes=None):
        """
        Initialize a Classifier model.
        
        Args:
            opt: Options object with model configuration
            num_answer_classes: Number of answer classes (auto-detected from vocab if None)
        """
        self.opt = opt
        self.vocab = get_vocab(opt)
        
        # Get number of answer classes from vocab
        if num_answer_classes is None:
            num_answer_classes = len(self.vocab['answer_token_to_idx'])
        self.num_answer_classes = num_answer_classes
        
        # Build answer vocabulary mappings
        self.answer_idx_to_token = {v: k for k, v in self.vocab['answer_token_to_idx'].items()}
        
        if opt.load_checkpoint_path is not None:
            self.load_checkpoint(opt)
        else:
            print(f'| creating new classifier network ({num_answer_classes} answer classes)')
            self.net_params = self._get_net_params(self.opt)
            self.gnn, self.classifier = create_classifier_net(**self.net_params)
        
        self.gpu_ids = opt.gpu_ids
        self.criterion = nn.CrossEntropyLoss()
        
        # Move to GPU if available
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            self.gnn.cuda(opt.gpu_ids[0])
            self.classifier.cuda(opt.gpu_ids[0])

    def load_checkpoint(self, opt):
        """Load model from checkpoint"""
        load_path = opt.load_checkpoint_path
        logging.info(f'| loading classifier checkpoint from {load_path}')
        checkpoint = torch.load(load_path, map_location='cpu')
        
        self.net_params = checkpoint['net_params']
        self.gnn, self.classifier = create_classifier_net(**self.net_params)
        
        self.gnn.load_state_dict(checkpoint['gnn_state'])
        self.classifier.load_state_dict(checkpoint['classifier_state'])

    def save_checkpoint(self, save_path):
        """Save model checkpoint"""
        checkpoint = {
            'net_params': self.net_params,
            'gnn_state': self.gnn.cpu().state_dict(),
            'classifier_state': self.classifier.cpu().state_dict(),
            'classification_mode': True  # Flag to identify this as a classifier checkpoint
        }
        torch.save(checkpoint, save_path)
        
        # Move back to GPU if needed
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.gnn.cuda(self.gpu_ids[0])
            self.classifier.cuda(self.gpu_ids[0])

    def set_input(self, x, y=None, g_data=None, answers=None):
        """
        Set input data for the model.
        
        Args:
            x: Questions (not used in pure classification, but kept for compatibility)
            y: Programs (not used in classification mode)
            g_data: Graph data (Gs, Gt) - THIS IS WHAT WE USE
            answers: Answer indices (ground truth for training)
        """
        self.g_data = None
        if g_data is not None:
            # Handle both tuple and list formats
            if isinstance(g_data, (tuple, list)) and len(g_data) >= 2:
                batch_s = self._to_var(g_data[0])
                batch_t = self._to_var(g_data[1])
                self.g_data = (batch_s, batch_t)
            else:
                self.g_data = self._to_var(g_data)
        
        self.answers = None
        if answers is not None:
            self.answers = self._to_var(answers)

    def _gnn_forward(self, g_data):
        """
        Forward pass through GNN (GraphMatcher).
        
        Args:
            g_data: Tuple of (batch_s, batch_t) graph batches
        
        Returns:
            g_embd: Graph embedding tensor [batch_size, gembd_vec_dim]
        """
        if g_data is None:
            raise ValueError("g_data is required for classification mode")
        
        BATCH_S, BATCH_T = g_data
        x_s, edge_index_s, edge_attr_s, batch_s = BATCH_S.x, BATCH_S.edge_index, BATCH_S.edge_attr, BATCH_S.batch
        x_t, edge_index_t, edge_attr_t, batch_t = BATCH_T.x, BATCH_T.edge_index, BATCH_T.edge_attr, BATCH_T.batch
        
        result = self.gnn(x_s, edge_index_s, edge_attr_s, batch_s,
                          x_t, edge_index_t, edge_attr_t, batch_t)
        
        # Handle if GNN returns tuple (embedding, attention_matrix)
        if isinstance(result, tuple):
            g_embd, _ = result
        else:
            g_embd = result
        
        # Ensure 2D output
        if len(g_embd.shape) == 3:
            g_embd = torch.mean(g_embd, dim=1)
        
        return g_embd

    def forward(self):
        """
        Forward pass: compute logits for classification.
        
        Returns:
            logits: Raw logits [batch_size, num_classes]
        """
        g_embd = self._gnn_forward(self.g_data)
        logits = self.classifier(g_embd)
        return logits

    def supervised_forward(self):
        """
        Compute loss for supervised training.
        
        Returns:
            loss: Scalar loss value
        """
        assert self.answers is not None, 'Must set answers for supervised training'
        
        logits = self.forward()
        self.loss = self.criterion(logits, self.answers)
        self.logits = logits  # Store for prediction
        
        return self._to_numpy(self.loss).item()

    def supervised_backward(self):
        """Backward pass for gradient computation"""
        assert self.loss is not None, 'Loss not defined, must call supervised_forward first'
        self.loss.backward()

    def predict(self):
        """
        Get predicted answer indices.
        
        Returns:
            predictions: Tensor of predicted answer indices [batch_size]
        """
        with torch.no_grad():
            logits = self.forward()
            predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu()

    def predict_answers(self):
        """
        Get predicted answer strings.
        
        Returns:
            List of predicted answer strings
        """
        predictions = self.predict()
        return [self.answer_idx_to_token.get(idx.item(), '<UNK>') for idx in predictions]

    def _get_net_params(self, opt):
        """Get network parameters from options"""
        net_params = {
            'gnn': opt.gnn,
            'gembd_vec_dim': opt.gembd_vec_dim,
            'is_padding_pos': opt.is_padding_pos,
            'num_answer_classes': self.num_answer_classes,
            'classifier_hidden_dim': getattr(opt, 'classifier_hidden_dim', 256),
            'classifier_dropout': getattr(opt, 'classifier_dropout', 0.1),
            'two_stream_gnn': getattr(opt, 'two_stream_gnn', 0),
        }
        return net_params

    def _to_var(self, x):
        """Convert tensor to appropriate device"""
        if x is None:
            return None
            
        device = 'cuda' if (len(self.gpu_ids) > 0 and torch.cuda.is_available()) else 'cpu'
        
        # Handle list of Batch objects (from dataloader collate)
        if isinstance(x, list):
            # If list contains Data/Batch objects, use Batch.from_data_list
            from torch_geometric.data import Batch, Data
            if len(x) > 0 and isinstance(x[0], Data):
                x = Batch.from_data_list(x)
            else:
                # Fallback: recursively convert each element
                return [self._to_var(elem) for elem in x]
        
        x = x.to(device)
        
        # Handle Batch objects from torch_geometric
        type_name = type(x).__name__
        if type_name in ['Batch', 'ClevrDataBatch']:
            for k in x.keys():
                var = x[k]
                if torch.is_tensor(var) and torch.is_floating_point(var):
                    var.requires_grad = True
            return x
        
        return Variable(x)

    def _to_numpy(self, x):
        """Convert tensor to numpy"""
        return x.data.cpu().numpy().astype(float)

    def parameters(self):
        """Get all trainable parameters"""
        return list(self.gnn.parameters()) + list(self.classifier.parameters())

    def train(self):
        """Set model to training mode"""
        self.gnn.train()
        self.classifier.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.gnn.eval()
        self.classifier.eval()

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  gnn={self.gnn},\n'
                f'  classifier={self.classifier}\n'
                f')')
