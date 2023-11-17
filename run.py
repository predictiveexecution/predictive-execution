from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import numpy as np
import random
from pathlib import Path
import time
import re
regex = '^[0-9]+$'
from transformers import (AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer)
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from dataset import CodeCoverageDataset
from model import Seq2Seq
from torch.nn.parallel import DataParallel
 
logger = logging.getLogger(__name__)        
 
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
 
def evaluate(args, eval_dataset, model, tokenizer):
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), 
                                 batch_size=args.eval_batch_size, drop_last=False)
    # Evaluate!
    logger.warning("***** Running evaluation *****")
    logger.warning(f"  Num examples = {len(eval_dataset)}")
    logger.warning(f"  Batch size = {args.eval_batch_size}")
    preds_pairs = []
    # Assuming 'model' is your original model
    model = DataParallel(model)
 
    model.eval()
    for batch in tqdm(eval_dataloader):
        # Move the batch to the GPU(s)
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            start_time = time.time()
            batch_preds = model(batch[0])
            end_time = time.time()
            prediction_time = end_time - start_time
            for i, item_preds in enumerate(batch_preds):
                text_topk = []
                for topk_pred in item_preds:
                    t = topk_pred.cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[: t.index(0)]
                    text_topk.append(tokenizer.decode(t, clean_up_tokenization_spaces=False))
                trace_preds = [int(x[1:-1]) for x in text_topk[0].split(' ') \
                    if ((re.search(regex, x[1:-1])) and (x.startswith('<')) and (x.endswith('>')))]
                print(text_topk)
                preds_pairs.append({'preds_topK': trace_preds, 'time' : prediction_time, 'trace' : text_topk})
    return preds_pairs
 
def main():
    parser = argparse.ArgumentParser()
 
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input/data caching path")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
 
    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--max_source_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_target_size", default=512, type=int,
                        help="Optional output sequence length after tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
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
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="beam size for beam search")
 
    args = parser.parse_args()
 
    # Setup CUDA, GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
 
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args.log_file = Path(args.output_dir) / 'log.txt'
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if Path(args.log_file).is_file():
        logfile = logging.FileHandler(args.log_file, 'a')
    else:
        logfile = logging.FileHandler(args.log_file, 'w')
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%m/%d/%Y %H:%M:%S %p')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.warning(f"Device: {device}, n_gpu: {args.n_gpu}")
 
    # Set seed
    set_seed(args)
 
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codeexecutor')
    special_tokens_list = ['<line>', '<state>', '</state>', '<dictsep>', '<output>', '<indent>',
                            '<dedent>', '<mask0>']
    for i in range(200):
        special_tokens_list.append(f"<{i}>")
    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict)
    config = RobertaConfig.from_pretrained('microsoft/codeexecutor')
 
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained('microsoft/codeexecutor', config=config)
    encoder.resize_token_embeddings(len(tokenizer))
    decoder = encoder
 
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=args.beam_size,
                    max_length=args.max_target_size, sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
                    eos_id=tokenizer.sep_token_id)
    model.to(args.device)
 
    logger.warning(f"Training/evaluation parameters {args}")
    eval_dataset = CodeCoverageDataset(tokenizer, './dataset/codenetmut_updated.json', args, logger)
    eval_results = evaluate(args, eval_dataset, model, tokenizer)
    with open('output/output_codeExecutor.json', 'w') as json_file:
        json.dump(eval_results, json_file, default=str)
    return eval_results
if __name__ == "__main__":
    main()