from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup, pipeline
from custom_dataset import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
import torch
import os, sys
import numpy as np
import argparse
import random
from src import UdpComms as U
import time

class Manager():
    def __init__(self, args):
        self.args = args
        
        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        special_tokens = {
            'bos_token': self.args.bos_token,
            'additional_special_tokens': [self.args.sp1_token, self.args.sp2_token]
        }
        self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        
        # Load model    
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.resize_token_embeddings(self.args.vocab_size)
        
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)
            
        if self.args.mode == 'train':            
            # Load optimizer
            print("Loading the optimizer...")
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
            self.best_loss = sys.float_info.max
            self.last_epoch = 0
            
            # Load train & valid dataset
            print("Loading train & valid data...")
            train_set = CustomDataset(self.args.train_prefix, self.args)
            valid_set = CustomDataset(self.args.valid_prefix, self.args)
            ppd = PadCollate(eos_id=self.args.eos_id)
            
            self.train_loader = DataLoader(train_set, 
                                           collate_fn=ppd.pad_collate, 
                                           shuffle=True, 
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
            self.valid_loader = DataLoader(valid_set, 
                                           collate_fn=ppd.pad_collate,
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
            
            if not os.path.exists(self.args.ckpt_dir):
                os.makedirs(self.args.ckpt_dir)
                
            # Calculate total training steps
            num_batches = len(self.train_loader)
            args.total_train_steps = args.num_epochs * num_batches
            args.warmup_steps = int(args.warmup_ratio * args.total_train_steps)
            
            self.sched = get_polynomial_decay_schedule_with_warmup(
                self.optim,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.total_train_steps,
                power=2
            )
            
            self.writer = SummaryWriter()
        
        if self.args.ckpt_name is not None:
            ckpt_path = f"{self.args.ckpt_dir}/{self.args.ckpt_name}.ckpt"
            if os.path.exists(ckpt_path):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(ckpt_path, map_location=self.args.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                
                if self.args.mode == 'train':
                    print(f"The training restarts with the specified checkpoint: {self.args.ckpt_name}.ckpt.")
                    self.optim.load_state_dict(ckpt['optim_state_dict'])
                    self.sched.load_state_dict(ckpt['sched_state_dict'])
                    self.best_loss = ckpt['loss']
                    self.last_epoch = ckpt['epoch']
                else:
                    print("The inference will start with the specified checkpoint.")
            else:
                print(f"Cannot fine the specified checkpoint {ckpt_path}.")
                if self.args.mode == 'train':
                    print("Training will start with the initialized model.")
                else:
                    print("Cannot inference.")
                    exit()
              
        print("Setting finished.")


              
    def infer(self,sock):
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",return_all_scores=True)
        print("Let's start!")
        print(f"If you want to quit the conversation, please type \"{self.args.end_command}\".")
        self.model.eval()
        self.fix_seed(self.args.seed)
        
        with torch.no_grad():
            input_hists = []
            sock.SendData("Hello! Stranger?")
            while True:
                fromunity = sock.ReadReceivedData()
                utter = input("You: ")
                if fromunity == self.args.end_command:
                    print("Bot: Good bye.")
                    break
                
                input_ids = [self.args.sp1_id] + self.tokenizer.encode(fromunity)
                input_hists.append(input_ids)
                
                if len(input_hists) >= self.args.max_turns:
                    num_exceeded = len(input_hists) - self.args.max_turns + 1
                    input_hists = input_hists[num_exceeded:]
                    
                input_ids = [self.args.bos_id] + list(chain.from_iterable(input_hists)) + [self.args.sp2_id]
                start_sp_id = input_hists[0][0]
                next_sp_id = self.args.sp1_id if start_sp_id == self.args.sp2_id else self.args.sp2_id
                assert start_sp_id != next_sp_id
                token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)]
                assert len(token_type_ids) == len(input_hists)
                token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [self.args.sp2_id]
                assert len(input_ids) == len(token_type_ids)
                input_len = len(input_ids)
                
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.args.device)
                token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.args.device)
                
                output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)                
                # output_ids = self.model.generate(
                #     input_ids=input_ids, token_type_ids=token_type_ids, pad_token_id=self.args.eos_id,
                #     do_sample=True, top_p=self.args.top_p, max_length=self.args.max_len,
                #     output_hidden_states=True, output_scores=True, return_dict_in_generate=True,
                # ).sequences
                # output_ids = output_ids[0].tolist()[input_len:]
                res = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                emotion=self.emotion_chage(res,classifier)
                sock.SendData(str(emotion)+str(res))
                print(f"Bot: {res}, {emotion}")
                input_hists.append([self.args.sp2_id] + self.tokenizer.encode(res))
                time.sleep(2)
                
    def nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, pos-1]  # (1, V)
            output = F.softmax(output, dim=-1)  # (1, V)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > self.args.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, V)
            
            probs = torch.zeros(output.shape, device=self.args.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1)  # (1, 1)
            
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)
            
            if idx_item == self.args.eos_id:
                break
                
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.args.sp2_id]]).to(self.args.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape
            
        return output_ids
    
    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    def emotion_chage(self, message,classifier):
        emotionlabel={'anger':'0','disgust':'1','fear':'2','joy':'3','neutral':'4','sadness':'5','surprise':'6'}
        max = 0
        maxlabel = ''
        for i in classifier(message)[0]:
            if i['score'] > max:
                max = i['score']
                maxlabel = i['label']
        return emotionlabel[maxlabel]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--mode', type=str, default="infer", help="The running mode: train or inference?")
    parser.add_argument('--data_dir', type=str, default="data", help="The name of the parent directory where data files are stored.")
    #parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of the train data files' name.")
    #parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of the validation data files' name.")
    parser.add_argument('--model_type', type=str, default="gpt2", help="The model type of GPT-2.")
    parser.add_argument('--bos_token', type=str, default="<bos>", help="The BOS token.")
    parser.add_argument('--sp1_token', type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument('--sp2_token', type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument('--gpu', type=str, default="0", help="The index of GPU to use.")
    parser.add_argument('--lr', type=float, default=2e-5, help="The learning rate.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument('--batch_size', type=int, default=8, help="The batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--num_epochs', type=int, default=10, help="The number of total epochs.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument('--max_turns', type=int, default=5, help="The maximum number of dialogue histories to include.")
    parser.add_argument('--top_p', type=float, default=0.9, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument('--ckpt_dir', type=str, default="saved_models", help="The directory name for saved checkpoints.")
    parser.add_argument('--ckpt_name', type=str, default="best_ckpt_epoch=6_valid_loss=2.6372", help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--end_command', type=str, default="bye!", help="The command to stop the conversation when inferencing.")
              
    args = parser.parse_args()
    sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
    manager = Manager(args)
    manager.infer(sock)
