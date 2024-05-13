from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os
import pickle
import random
import re
import numpy as np
import torch
import torch.nn as nn
from graphcodebert_model import Seq2Seq
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
import pandas as pd

cpu_cont = 16
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids):
        self.input_ids = input_ids


class TextDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.examples = []
        df = pd.read_csv(args.test_data_file)
        sources = df["source"].tolist()
        for i in tqdm(range(len(sources))):
            self.examples.append(convert_examples_to_features(sources[i], tokenizer, args))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].input_ids, self.examples[i].input_ids.ne(1)

def convert_examples_to_features(source, tokenizer, args):
    # encode
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length',
                                  return_tensors='pt')
    return InputFeatures(source_ids)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def compare_strings_ignore_whitespace(str1, str2):
    # 移除字符串中的空格
    str1_without_space = re.sub(r'\s', '', str1)
    str2_without_space = re.sub(r'\s', '', str2)

    # 使用re.match()函数进行匹配比较
    if re.match(f'^{re.escape(str1_without_space)}$', str2_without_space):
        return True
    else:
        return False

def eval(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    raw_predictions = []
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        (input_ids, attention_mask) = [x.squeeze(1).to(args.device) for x in batch]
        with torch.no_grad():
            beam_outputs = model(source_ids=input_ids, source_mask=attention_mask)
            beam_outputs = beam_outputs.detach().cpu().tolist()

        for single_output in beam_outputs:
            # pred
            for one_beam in single_output:
                prediction = tokenizer.decode(one_beam, skip_special_tokens=False)
                raw_predictions.append(clean_tokens(prediction))

    df = pd.read_csv(args.test_data_file)
    df['assert_pred'] = raw_predictions
    targets = df['target']
    match = []
    for p,t in zip(raw_predictions,targets):
        if compare_strings_ignore_whitespace(p,t):
            match.append(1)
        else:
            match.append(0)
    if not os.path.exists(args.result_output_dir):
        os.makedirs(args.result_output_dir)
    df['match'] = match
    df.drop('source',axis=1).to_csv(os.path.join(args.result_output_dir, 'assertion_preds.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--result_output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--beam_size", default=50, type=int,
                        help="Beam size to use when decoding.")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                        help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model_from_checkpoint", action='store_true', default=False,
                        help="Whether to load model from checkpoint.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")
    parser.add_argument("--output_name", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--n_gpu', type=int, default=1,
                        help="using which gpu")
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    # build model
    encoder = RobertaModel.from_pretrained("microsoft/graphcodebert-base", config=config)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    encoder.resize_token_embeddings(len(tokenizer))
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.decoder_block_size,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    model.to(device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir, map_location=args.device))
    test_dataset = TextDataset(tokenizer, args)
    eval(args, model, tokenizer, test_dataset, best_threshold=0.5)


if __name__ == "__main__":
    main()
