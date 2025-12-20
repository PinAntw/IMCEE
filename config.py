#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Research/IMCEE/config.py

import os


# ============================================================
# Configuration
# ============================================================
class PathConfig:
    BASE_DIR = "/home/joung/r13725060/Research/IMCEE"
    EXP_DIR  = os.path.join(BASE_DIR, "data/Vexplain")
    EXP_DIR_GPT  = os.path.join(BASE_DIR, "data/Vexplain_gpt")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_tower")

    # 用於存放 LLM 文字解釋與標籤的路徑
    GPT_TEXT_DIR = os.path.join(BASE_DIR, "data/plaintext_gpt4omini")

    # Filename tag
    SAVE_SUFFIX = "expandData_concat_layer5_earlyfusionNode_type3add_large"
    DATA_DIR = os.path.join(BASE_DIR, "data/preprocess")

    CONV_PATH = os.path.join(DATA_DIR, "conversations.jsonl")

    NEW_DATA_DIR = os.path.join(BASE_DIR, "data/preprocess_expanded_final")
    TRAIN_PATH = os.path.join(NEW_DATA_DIR, "new_pairs_train.jsonl")
    VALID_PATH = os.path.join(NEW_DATA_DIR, "new_pairs_valid.jsonl")
    TEST_PATH  = os.path.join(NEW_DATA_DIR, "new_pairs_test.jsonl")

    LLM_SHORTCUT_TRAIN = os.path.join(GPT_TEXT_DIR, "explain_train_gpt4omini.jsonl")
    LLM_SHORTCUT_VALID = os.path.join(GPT_TEXT_DIR, "explain_valid_gpt4omini.jsonl")
    LLM_SHORTCUT_TEST  = os.path.join(GPT_TEXT_DIR, "explain_test_gpt4omini.jsonl")

    EX_TRAIN_PT  = os.path.join(EXP_DIR, "explain_train_embeddings.pt")
    EX_TRAIN_TSV = os.path.join(EXP_DIR, "explain_train_results_index.tsv")
    EX_VALID_PT  = os.path.join(EXP_DIR, "explain_valid_embeddings.pt")
    EX_VALID_TSV = os.path.join(EXP_DIR, "explain_valid_results_index.tsv")
    EX_TEST_PT   = os.path.join(EXP_DIR, "explain_test_embeddings.pt")
    EX_TEST_TSV  = os.path.join(EXP_DIR, "explain_test_results_index.tsv")

    ES_TRAIN_PT  = os.path.join(EXP_DIR_GPT, "explain_train_embeddings.pt")
    ES_TRAIN_TSV = os.path.join(EXP_DIR_GPT, "explain_train_gpt4omini_index.tsv")
    ES_VALID_PT  = os.path.join(EXP_DIR_GPT, "explain_valid_embeddings.pt")
    ES_VALID_TSV = os.path.join(EXP_DIR_GPT, "explain_valid_gpt4omini_index.tsv")
    ES_TEST_PT   = os.path.join(EXP_DIR_GPT, "explain_test_embeddings.pt")
    ES_TEST_TSV  = os.path.join(EXP_DIR_GPT, "explain_test_gpt4omini_index.tsv")

    @property
    def SAVE_MODEL_PATH(self):
        return os.path.join(self.CKPT_DIR, f"model_{self.SAVE_SUFFIX}.pt")

    @property
    def OUT_JSON_PATH(self):
        return os.path.join(self.BASE_DIR, "outputs", f"test_result_{self.SAVE_SUFFIX}.json")

    @property
    def TENSORBOARD_LOG_DIR(self):
        return os.path.join(self.BASE_DIR, "outputs", "runs")


class ModelConfig:
    # TEXT_MODEL = "roberta-base"
    # HIDDEN_DIM = 768
    TEXT_MODEL = "roberta-large"
    HIDDEN_DIM = 1024
    FREEZE_TEXT = True
    NUM_GNN_LAYERS = 5
    GNN_DROPOUT    = 0.3
    BASE_DROPOUT   = 0.2
    USE_EXPLAIN = False
    USE_EXPLAIN_SPACE = True


class TrainConfig:
    RUN_MODE = "train"   # "train" or "test"
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 4
    ACCUM_STEPS = 1
    PATIENCE = 8
    WARMUP_RATIO = 0.1
    POS_WEIGHT_MULT = 1
    LAMBDA_EXPL = 1.0


class OptimConfig:
    LR_BASE = 1e-4
    WD_BASE = 1e-4
    LR_PLM  = 3e-6
    WD_PLM  = 0.01
    LR_GNN  = 1e-3
    WD_GNN  = 0.0


class Config:
    path = PathConfig()
    model = ModelConfig()
    train = TrainConfig()
    optim = OptimConfig()


cfg = Config()
