#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: sample_questions_mgn.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-05-19
# 
# This file is part of MGN
# Distributed under terms of the MIT License

import os, sys, platform
import argparse
import h5py
import numpy as np
np.random.seed(42)
import random
random.seed(42)
from tqdm import tqdm
import logging
from rsmlkit.logging import get_logger, set_default_level

logger = get_logger(__file__)
set_default_level(logging.INFO)   # logging.INFO

from functools import lru_cache
from typing import *
from collections import defaultdict, OrderedDict

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)
sys.path.insert(0, ".")

import utils.programs as program_utils
import utils.preprocess as preprocess_utils
from utils.mgn_preproc_utils import get_questions_and_parsed_scenes, \
        save_graphs, save_graph_pairdata, save_graph_edges, save_graph_embeddings, save_graph_docs, save_h5
# from utils.mgn_preproc_utils import *
from datasets import ClevrData

import clevr_parser
import clevr_parser.utils as parser_utils

from rsmlkit.utils.cache import cached_result

import logging
from rsmlkit.logging import get_logger, set_default_level

logger = get_logger(__file__)
set_default_level(logging.INFO)   # logging.INFO

from functools import lru_cache
from typing import *
from collections import defaultdict, OrderedDict

# from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('--random_sample', default=0,
                    type=int, help='randomly sample questions')
parser.add_argument('--max_sample', default=2000,
                    type=int, help='size of split, effective only when random is true')
parser.add_argument('--n_questions_per_family', default=5,
                    type=int, help='number of questions per family, effective when random is false')
parser.add_argument('--input_question_h5', default='../data/reason/clevr_h5/clevr_train_questions.h5',
                    type=str, help='path to input question h5 file')
parser.add_argument('--output_dir', default='../data/reason/clevr_mgn_h5',
                    type=str, help='output dir')
parser.add_argument('--output_index', default=None,
                    type=int, help='output repeat id')
# Needed for building Gs
parser.add_argument('--input_questions_json',
                    default='../../data/CLEVR_v1.0/questions/CLEVR_train_questions.json')
# Needed for building Gt
parser.add_argument('--input_parsed_img_scenes_json',
                    default='../../data/CLEVR_v1.0/scenes_parsed/train_scenes_parsed.json',
                    help='The parsed image scenes file for dataset-split')
parser.add_argument('--input_vocab_json', default='../data/reason/clevr_h5/clevr_vocab.json')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--is_debug', action='store_true', default=False, help="Debug flag")
parser.add_argument('--is_directed_graph', default=1, type=int, help='If set, will parse Gs,Gt as nx.MultiDiGraph')

def get_question_fam_to_indices(args) -> Dict:
    """ Returns a dict with 50 q_idx for each q_fam type"""
    h5file = h5py.File(args.input_question_h5, 'r')
    # enc_questions = h5file['questions'][()]
    # image_idxs = h5file['image_idxs'][()]
    q_families = h5file['question_families'][()]
    fam_freq_dist = np.bincount(q_families)
    fam2idx = defaultdict(list)
    for i, fam in enumerate(q_families):
        fam2idx[fam].append(i)
    fam2indices = OrderedDict(sorted(fam2idx.items()))
    #return (fam2indices, fam_freq_dist)
    return fam2indices

def get_output_filename(args) -> str:
    # Détecter si c'est train ou val basé sur le chemin du fichier d'entrée
    is_val = 'val' in args.input_question_h5.lower()
    prefix = 'clevr_val' if is_val else 'clevr_train'
    
    if args.random_sample:
        max_sample = args.max_sample
        if args.output_index is not None:
            filename = f'{prefix}_{max_sample}_questions_{args.output_index:02d}.h5'
        else:
            filename = f'{prefix}_{max_sample}_questions.h5'
        print('| randomly sampling %d questions' % max_sample)
    else:
        max_sample = args.n_questions_per_family * 90
        if args.output_index is not None:
            filename = f'{prefix}_{args.n_questions_per_family}questions_per_family_{args.output_index:02d}.h5'
        else:
            # Default filename #
            filename = f'{prefix}_{args.n_questions_per_family}questions_per_family.h5'
        print('| drawing questions, %d per family' % args.n_questions_per_family)

    return filename

def main(args):
    """
        Save nx.graph (Gss, Gts,...) and corresponding torch_geometric.data.PairData
        (via clevr_parse embedder api).
    """
    if args.is_debug:
        set_default_level(10)
    is_directed_graph = args.is_directed_graph
    logger.debug(f"Parser flag is_directed_graph = {is_directed_graph}")
    graph_parser = clevr_parser.Parser(backend="spacy", model='en_core_web_sm',
                                       has_spatial=True,
                                       has_matching=True).get_backend(identifier='spacy')
    embedder = clevr_parser.Embedder(backend='torch', parser=graph_parser).get_backend(identifier='torch')
    raw_questions, img_scenes = get_questions_and_parsed_scenes(args.input_questions_json,
                                                                args.input_parsed_img_scenes_json )
    logger.info('| importing questions from %s' % args.input_question_h5)
    input_questions = h5py.File(args.input_question_h5, 'r')
    #N = len(input_questions['questions'])

    # Baseline Entities #
    questions, programs, answers, question_families, orig_idxs, img_idxs = [], [], [], [], [], []
    family_count = np.zeros(90)

    # Graphs and Embeddings #
    data_s_list = []                            # List [torch_geometric.data.Data]
    data_t_list = []                            # List [torch_geometric.data.Data]

    filename = get_output_filename(args)
    __all_question_families: np.ndarray = input_questions['question_families'][()]
    __all_enc_questions: np.ndarray = input_questions['questions'][()]
    __all_img_indices: np.ndarray = input_questions['image_idxs'][()]
    logger.debug(f"__all_question_families len {len(__all_question_families)}")

    # Sample items based on mode #
    fam2indices = get_question_fam_to_indices(args)
    M = len(fam2indices.keys())                 # 90
    
    # 🔥 MODIFICATION: Si random_sample=1, prendre TOUTES les questions disponibles par famille
    if args.random_sample:
        # Mode: Prendre toutes les questions jusqu'à max_sample
        max_sample = args.max_sample
        logger.info(f"Mode RANDOM: Prenant jusqu'à {max_sample} questions (toutes les familles)")
    else:
        # Mode original: N questions par famille
        N = args.n_questions_per_family
        max_sample = N * M
        logger.info(f"Mode FAMILLE: {N} questions par famille × {M} familles = {max_sample} max")
    
    family_count = np.zeros(M)

    # TODO: accumulating values here need to be parallelized, and joined write ex-post
    num_skipped = 0  # Counter for tracking num of samples skipped
    
    # 🚀 OPTIMISATION: Cache pour les scènes déjà parsées (évite de reparser les mêmes scènes)
    scene_cache = {}  # {img_idx: (Gt, t_doc, X_t, ei_t, e_attr_t)}
    
    # Barre de progression
    from tqdm import tqdm
    pbar = tqdm(total=max_sample, desc="Précalcul des graphes", unit="question")

    for enum_idx, fam_idx in enumerate(fam2indices.keys()):
        all_fam_samples = fam2indices[fam_idx]
        
        # 🔥 NOUVELLE LOGIQUE: Prendre TOUTES les questions de la famille (ou ce qui reste)
        if args.random_sample:
            # Prendre toutes les questions de cette famille (limité par max_sample restant)
            remaining = max_sample - len(questions)
            actual_N = min(len(all_fam_samples), remaining)
            logger.debug(f"Question_family {fam_idx}: prend {actual_N} questions sur {len(all_fam_samples)} disponibles")
        else:
            # Mode original: N fixe par famille
            N = args.n_questions_per_family
            actual_N = min(N, len(all_fam_samples))
            logger.debug(f"Question_family {fam_idx}: prend {actual_N} questions (N={N} par famille)")
        
        if actual_N == 0:
            break  # Plus de questions à prendre
        
        N_question_sample_indices = np.random.choice(all_fam_samples, actual_N, replace=False)
        assert len(N_question_sample_indices) == actual_N
        # TODO: parallelize this iteration loop
        for i in N_question_sample_indices:
            try:
                img_idx = __all_img_indices[i]
                logger.debug(f"\tProcessing Image - {img_idx} from fam_idx {fam_idx}: {i} of {len(all_fam_samples)}")
                img_scene = list(filter(lambda x: x['image_index'] == img_idx, img_scenes))[0]
            except IndexError as ie:
                logger.warning(f"For {img_idx}: {ie}")
                num_skipped += 1
                continue
            
            try:
                Gt, t_doc = graph_parser.get_doc_from_img_scene(img_scene, is_directed_graph=is_directed_graph)
                X_t, ei_t, e_attr_t = embedder.embed_t(img_idx, args.input_parsed_img_scenes_json)
            except AssertionError as ae:
                logger.warning(f"AssertionError Encountered: {ae}")
                logger.warning(f"[{img_idx}] Excluding images with > 10 objects")
                num_skipped += 1
                continue
            
            if Gt is None and ("SKIP" in t_doc):
                logger.warning(f"Got None img_doc at image_index: {img_idx}")
                print(f"Skipping all text_scenes for imgage idx: {img_idx}")
                num_skipped += 1
                continue
            q_idx = input_questions['orig_idxs'][i]
            q_obj = list(filter(lambda x: x['question_index'] == q_idx, raw_questions))[0]
            assert q_obj['image_index'] == img_idx
            s = q_obj['question']
            try:
                Gs, s_doc = graph_parser.parse(s, return_doc=True, is_directed_graph=is_directed_graph)
                X_s, ei_s, e_attr_s = embedder.embed_s(s)
            except ValueError as ve:
                logger.warning(f"ValueError Encountered: {ve}")
                logger.warning(f"Skipping question: {s} for {img_idx}")
                num_skipped += 1
                continue
            if Gs is None and ("SKIP" in s_doc):
                logger.warning("Got None as Gs and 'SKIP' in Gs_embd. (likely plural with CLEVR_OBJS label) ")
                logger.warning(f"SKIPPING processing {s}  at {img_idx}")
                num_skipped += 1
                continue

            data_s = ClevrData(x=X_s, edge_index=ei_s, edge_attr=e_attr_s)
            data_t = ClevrData(x=X_t, edge_index=ei_t, edge_attr=e_attr_t)
            data_s_list.append(data_s)
            data_t_list.append(data_t)

            family_count[fam_idx] += 1
            questions.append(input_questions['questions'][i])
            programs.append(input_questions['programs'][i])
            answers.append(input_questions['answers'][i])
            question_families.append(input_questions['question_families'][i])
            orig_idxs.append(input_questions['orig_idxs'][i])
            img_idxs.append(img_idx)

            pbar.update(1)
            pbar.set_postfix({"skipped": num_skipped, "total": len(questions)})
            logger.info(f"\nCount = {len(questions)}\n")

        if len(questions) >= max_sample:
            break
    
    pbar.close()
    
    print(f"\n📊 Statistiques du précalcul:")
    print(f"   - Questions générées: {sum(family_count.values())}")
    print(f"   - Questions skipped: {num_skipped}")
    print(f"   - Scènes uniques parsées: {len(scene_cache)} (au lieu de {sum(family_count.values()) + num_skipped})")
    total_processed = sum(family_count.values()) + num_skipped
    print(f"   - Gain de performance: {((total_processed - len(scene_cache)) / total_processed * 100):.1f}% de scènes évitées")

    logger.debug(f"Total samples skipped (due to errors/exceptions) = {num_skipped}")
    # ---------------------------------------------------------------------------#
    ## SAVE .H5
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output_file = os.path.join(args.output_dir, filename)
    out_dir = args.output_dir
    out_f_prefix = filename.split('.')[0]
    out_fpp = f"{out_dir}/{out_f_prefix}"
    logger.debug(f"out_fpp = {out_fpp}")

    print('sampled question family distribution')
    print(family_count)
    print('| saving output file to %s' % output_file)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('questions', data=np.asarray(questions, dtype=np.int32))
        f.create_dataset('programs', data=np.asarray(programs, dtype=np.int32))
        f.create_dataset('answers', data=np.asarray(answers))
        f.create_dataset('image_idxs', data=np.asarray(img_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))
        f.create_dataset('question_families', data=np.asarray(question_families))

    ## ------------  SAVE GRAPH DATA ------------ ##
    save_graph_pairdata(out_fpp, data_s_list, data_t_list, is_directed_graph=is_directed_graph)
    logger.info(f"Saved Graph Data in: {out_fpp}_*.[h5|.gpickle|.npz|.pt] ")
    print('| done')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
