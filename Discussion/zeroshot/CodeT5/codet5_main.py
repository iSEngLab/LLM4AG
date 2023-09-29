from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          T5ForConditionalGeneration, RobertaTokenizer, T5Config)
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

cpu_cont = 16
logger = logging.getLogger(__name__)
templates = ["assertFalse", "assertEquals", "assertTrue",  "assertNull", "assertNotNull", "assertThat", "assertSame",
             "assertNotSame"]


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 label,
                 decoder_input_ids):
        self.input_ids = input_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        sources = df["source"].tolist()
        labels = df["target"].tolist()
        for i in tqdm(range(len(sources))):
            self.examples.append(convert_examples_to_features(sources[i], labels[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("decoder_input_ids: {}".format(' '.join(map(str, example.decoder_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].input_ids, self.examples[i].input_ids.ne(0), self.examples[i].label, self.examples[
            i].decoder_input_ids

def removeSub(s,subs):
    parts = s.split(subs, 1)

    # 如果分割成功，取分割后的第一部分即可
    if len(parts) > 1:
        result = parts[0]
    else:
        result = s
    return result
def convert_examples_to_features(source, label, tokenizer, args):
    source = removeSub(source,'"<FocalMethod>"')
    # encode
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length',
                                  return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size,
                                         padding='max_length', return_tensors='pt')
    label = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length',
                             return_tensors='pt')
    return InputFeatures(source_ids, label, decoder_input_ids)


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


def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    accuracy = []
    raw_predictions = []
    correct_prediction = ""
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        correct_pred = False
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
        original_code = clean_tokens(tokenizer.decode(input_ids[0], skip_special_tokens=False))
        ground_truth = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
        ground_truth = clean_tokens(ground_truth)
        tem = []
        for template in templates:
            if template in ground_truth:
                to_replace = template + " ( <extra_id_0> )"
                code_with_template = original_code.replace('"<AssertPlaceHolder>"',to_replace)
                t_input = tokenizer.encode(code_with_template, truncation=True, max_length=args.encoder_block_size, padding='max_length',
                                      return_tensors='pt').to(args.device)
                attention_mask = t_input.ne(0).to(args.device)
                with torch.no_grad():
                    beam_outputs = model.generate(input_ids=t_input,
                                                  attention_mask=attention_mask,
                                                  do_sample=False,  # disable sampling to test if batching affects output
                                                  num_beams=args.num_beams,
                                                  num_return_sequences=args.num_beams,
                                                  max_length=args.decoder_block_size)
                beam_outputs = beam_outputs.detach().cpu().tolist()
                for single_output in beam_outputs:
                    # pred
                    prediction = tokenizer.decode(single_output, skip_special_tokens=True)
                    prediction = clean_tokens(prediction)
                    prediction = template + " ( "+prediction+" )"
                    # truth
                    tem.append(prediction)
                    if re.sub(r'\s', '', prediction) == re.sub(r'\s', '', ground_truth):
                        correct_pred = True
        if correct_pred:
            accuracy.append(1)
        else:
            accuracy.append(0)
        raw_predictions.append(tem)
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")
    df = pd.read_csv(args.test_data_file)
    source = np.array(df["source"]).tolist()
    target = np.array(df["target"]).tolist()
    f = open('../data/raw_predictions/CodeT5/CodeT5_{}.txt'.format(args.output_name), 'w', encoding='utf-8')
    for s, t, r, a in zip(source, target, raw_predictions, accuracy):
        f.write("source:\n" + s + "\n")
        f.write("target:\n" + t + "\n")
        f.write("match:\n" + str(a) + "\n")
        f.write("raw_predictions:\n")
        for i in r:
            f.write(i + "\n")


def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
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
    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                        help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model_from_checkpoint", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
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
    parser.add_argument('--n_gpu', type=int, default=2,
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
    # tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    # model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    model.to(device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_test:
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)


if __name__ == "__main__":
    main()
