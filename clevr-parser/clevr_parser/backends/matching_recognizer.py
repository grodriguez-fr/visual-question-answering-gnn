#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File   : matching_recognizer.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/21/2019
#
# This file is part of CLEVR Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

from __future__ import unicode_literals, print_function

import logging
import os

from clevr_parser import setup_logging
from clevr_parser.utils import *
from clevr_parser.config import config_entity_ruler
from spacy.pipeline import EntityRuler
from spacy.language import Language

logger = setup_logging(__name__, log_level=logging.DEBUG)

__all__ = ['MatchingRecognizer']

class MatchingRecognizer(object):

    name = "matching_recognizer"

    def __init__(self, nlp, label="MATCHING_RE", **kwargs):
        self.label = nlp.vocab.strings[label]
        self.nlp = nlp
        
        # self.ruler = EntityRuler(nlp, overwrite_ents=False, validate=True)
        ruler_name = f"entity_ruler_{self.name}"
        # ruler_name = f"{self.name}"  # will create cyclic loop
        self.ruler = EntityRuler(nlp, name=ruler_name, **config_entity_ruler)

        self.is_debug = kwargs.get("is_debug")
        try:
            # Use absolute path based on file location
            file_path = os.path.join(os.path.dirname(__file__), '..', '_data', 'matching-re-patterns.jsonl')
            self.ruler = self.ruler.from_disk(file_path)
            self._add_ruler_to_pipeline(nlp, self.ruler)
        except ValueError as ve:
            if self.is_debug:
                logger.error(f"{ve}: Ensure patterns file is added.")
            self._add_patterns()
        if self.is_debug:
            logger.debug(f"Pipeline -> {nlp.pipe_names}")

    def __call__(self, doc):
        #preprocess
        return doc

    def construct_patters(self):
        """Load patterns adhoc if loading .jsonl file fails"""
        obj_patterns = [
            {"label": "MATCHING_RE", "pattern": [{"LOWER": "is"}, {"LOWER": "the"}, {"LOWER": "same"},
                                                 {"TEXT": {"IN": ["size", "color", "material", "shape"]}}]},
            {"label": "MATCHING_RE", "pattern": [{"LOWER": "are"}, {"LOWER": "the"}, {"LOWER": "same"},
                                                 {"TEXT": {"IN": ["size", "color", "material", "shape"]}}]},
            {"label": "MATCHING_RE", "pattern": [{"LOWER": "has"}, {"LOWER": "the"}, {"LOWER": "same"},
                                                 {"TEXT": {"IN": ["size", "color", "material", "shape"]}}]},
            {"label": "MATCHING_RE", "pattern": [{"LOWER": "same"},
                                                 {"TEXT": {"IN": ["size", "color", "material", "shape"]}}]}
        ]
        return obj_patterns

    def _add_patterns(self):
        patterns = self.construct_patters()
        #other_pipes = [p for p in self.nlp.pipe_names if p != "tagger"] # excluse tagger
        other_pipes = self.nlp.pipe_names
        with self.nlp.disable_pipes(*other_pipes):
            self.ruler.add_patterns(patterns)

        self._add_ruler_to_pipeline(self.nlp, self.ruler)

    # def _add_ruler_to_pipeline(self, nlp, ruler, name="matching_entity_ruler"):
    def _add_ruler_to_pipeline(self, nlp, ruler, name="entity_ruler_matching_recognizer"):
        pipeline, _ = zip(*nlp.pipeline) if nlp.pipeline else ([], [])
        if name in pipeline:
            idx = pipeline.index(name)
            nlp._components[idx] = (name, ruler)
        else:
            nlp._components.append((name, ruler))



# Register the custom component for spaCy 3.x
if not Language.has_factory("matching_recognizer"):
    @Language.factory("matching_recognizer")
    def create_matching_recognizer(nlp, name):
        return MatchingRecognizer(nlp)