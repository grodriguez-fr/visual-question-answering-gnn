from .encoder import Encoder
from .decoder import Decoder
from .gnn import GIN, MLP, GAT
from .graph_matcher import GraphMatcher
from .classification_head import ClassificationHead

from .seq2seq import Seq2seq
from .seq2seq_gnn import Seq2seqGNN
import utils.utils as utils

import torch
import torch.nn as nn

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

def get_vocab(opt):
    if opt.dataset == 'clevr':
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    vocab = utils.load_vocab(vocab_json)
    return vocab


def create_seq2seq_net(input_vocab_size, output_vocab_size, hidden_size, 
                       word_vec_dim, n_layers, bidirectional, variable_lengths, 
                       use_attention, encoder_max_len, decoder_max_len, start_id, 
                       end_id, word2vec_path=None, fix_embedding=False, gembd_vec_dim=0):
    word2vec = None
    if word2vec_path is not None:
        word2vec = utils.load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len, 
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding, gembd_vec_dim=gembd_vec_dim)
    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers, start_id, end_id,
                      bidirectional=bidirectional, use_attention=use_attention)

    return Seq2seq(encoder, decoder)


def create_seq2seq_gnn_net(input_vocab_size, output_vocab_size, hidden_size,
                       word_vec_dim, n_layers, bidirectional, variable_lengths,
                       use_attention, encoder_max_len, decoder_max_len, start_id,
                       end_id, gnn, gembd_vec_dim, is_padding_pos, word2vec_path=None, fix_embedding=False, **kwargs):
    """ Creates a Seq2seqGNN model for e2e training for Gs, Gt embedding """
    word2vec = None
    if word2vec_path is not None:
        word2vec = utils.load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len,
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding, gembd_vec_dim=gembd_vec_dim)
    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers, start_id, end_id,
                      bidirectional=bidirectional, use_attention=use_attention)

    logging.info(f"Instantiating GNN type: {gnn}")
    in_channels = gembd_vec_dim+3 if is_padding_pos else gembd_vec_dim      # 99 | 96    
    
    if gnn == 'gat':
        psi_1 = GAT(in_channels=in_channels, out_channels=gembd_vec_dim, num_layers=3, heads=4)
    elif gnn == 'gin':
        psi_1 = GIN(in_channels, gembd_vec_dim, num_layers=3)
    else:
        raise ValueError(f"Type de GNN inconnu : {gnn}")
    
    # In two-stream mode, create a second independent GNN for scenes
    psi_2 = None
    if kwargs.get('two_stream_gnn', 0):
        if gnn == 'gat':
            psi_2 = GAT(in_channels=in_channels, out_channels=gembd_vec_dim, num_layers=3, heads=4)
        elif gnn == 'gin':
            psi_2 = GIN(in_channels, gembd_vec_dim, num_layers=3)    
    # Create GraphMatcher with two_stream parameter
    gnn_matcher = GraphMatcher(psi_1, psi_2=psi_2, gembd_vec_dim=gembd_vec_dim, 
                               aggregation='cat', two_stream=kwargs.get('two_stream_gnn', 0))

    return Seq2seqGNN(encoder, decoder, gnn_matcher)


def create_classifier_net(gnn, gembd_vec_dim, is_padding_pos, num_answer_classes,
                          classifier_hidden_dim=256, classifier_dropout=0.1, **kwargs):
    """
    Creates a GNN + Classification Head model for end-to-end VQA.
    
    This model follows the paper architecture:
        Questions (LSTM) + Graphs (GNN) -> Encoder -> ClassificationHead -> answer
    
    Args:
        gnn (str): GNN type ('gin' or 'gat')
        gembd_vec_dim (int): Graph embedding dimension (default: 96)
        is_padding_pos (int): Whether to include 3D positions (adds 3 dims)
        num_answer_classes (int): Number of possible answers
        classifier_hidden_dim (int): Hidden dim for classification MLP
        classifier_dropout (float): Dropout for classification head
    
    Returns:
        tuple: (gnn_matcher, classifier_head)
    """
    logging.info(f"Creating Classifier Net with GNN type: {gnn}")
    in_channels = gembd_vec_dim + 3 if is_padding_pos else gembd_vec_dim  # 99 | 96
    
    if gnn == 'gat':
        psi_1 = GAT(in_channels=in_channels, out_channels=gembd_vec_dim, num_layers=3, heads=4)
    elif gnn == 'gin':
        psi_1 = GIN(in_channels, gembd_vec_dim, num_layers=3)
    else:
        raise ValueError(f"Unknown GNN type: {gnn}")
    
    # In two-stream mode, create a second independent GNN for scenes
    psi_2 = None
    if kwargs.get('two_stream_gnn', 0):
        if gnn == 'gat':
            psi_2 = GAT(in_channels=in_channels, out_channels=gembd_vec_dim, num_layers=3, heads=4)
        elif gnn == 'gin':
            psi_2 = GIN(in_channels, gembd_vec_dim, num_layers=3)
    
    # Create GraphMatcher
    gnn_matcher = GraphMatcher(psi_1, psi_2=psi_2, gembd_vec_dim=gembd_vec_dim,
                               aggregation='cat', two_stream=kwargs.get('two_stream_gnn', 0))
    
    # 🔥 Classification Head input dimension = encoder hidden size
    # Si bidirectionnel: hidden_size * 2, sinon juste hidden_size
    encoder_hidden_size = kwargs.get('encoder_hidden_size', 256)
    bidirectional = kwargs.get('bidirectional_encoder', 1)
    classifier_input_dim = encoder_hidden_size * 2 if bidirectional else encoder_hidden_size
    
    # Create Classification Head
    classifier_head = ClassificationHead(
        input_dim=classifier_input_dim,  # 🔥 Maintenant basé sur l'encoder, pas le GNN
        hidden_dim=classifier_hidden_dim,
        num_classes=num_answer_classes,
        dropout=classifier_dropout
    )
    
    return gnn_matcher, classifier_head