import random
import pickle
from transformers import *
import torch
from torch.utils.data import DataLoader

tag2id = {'S':0, 'B':1, 'M':2, 'E':3}
id2tag = {0:'S', 1:'B', 2:'M', 3:'E'}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, qtok, tags):
        self.toks = qtok
        self.tags = tags

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, idx):
        tok, tag = self.toks[idx], self.tags[idx]
        return tok, tag

def custom_collate(batch):
    transposed = list(zip(*batch))
    lst = []
    # transposed[0]: list of token ids of a question
    padded_seq = []
    max_seq_len = len(max(transposed[0], key=len))
    for seq in transposed[0]:
        padded_seq.append(seq + [0] * (max_seq_len - len(seq)))
    lst.append(torch.LongTensor(padded_seq))
    # lst.append(padded_seq)

    # tansposed[1]: list of tag ids of SAME LENGTH!
    padded_tag = []
    att_mask = []
    for seq in transposed[1]:
        padded_tag.append(seq + [0] * (max_seq_len - len(seq)))
        att_mask.append([1]*len(seq) + [0]*(max_seq_len-len(seq)))
    lst.append(torch.LongTensor(padded_tag))
    # lst.append(padded_tag)
    lst.append(torch.FloatTensor(att_mask))
    # lst.append(att_mask)

    return lst


def load_dataset(tokenizer, train_batch, dev_batch, max_sent_len=256, extra_info=False):
    f = open('./data/train.txt', 'r', encoding='utf-8')
    t_ori = f.read().strip().split('\n')
    f.close()

    f = open('./data/test.answer.txt', 'r', encoding='utf-8')
    v_ori = f.read().strip().split('\n')
    f.close()

    t_ori = [a.strip().split('  ') for a in t_ori]
    v_ori = [a.strip().split('  ') for a in v_ori]

    t_ori_prun, v_ori_prun = [],[]

    # prun data set
    for ori in t_ori:
        ct = 0
        cur_ori = []
        for word in ori:
            cur_ori.append(word)
            ct += len(word)
            if ct>max_sent_len:
                t_ori_prun.append(cur_ori.copy())
                ct = 0
                cur_ori = []
        if len(cur_ori)==0:
            continue
        t_ori_prun.append(cur_ori.copy())

    extra_pos = []
    for ori in v_ori:
        ct = 0
        cur_ori = []
        for word in ori:
            cur_ori.append(word)
            ct += len(word)
            if ct>max_sent_len:
                v_ori_prun.append(cur_ori.copy())
                ct = 0
                cur_ori = []
                extra_pos.append(len(v_ori_prun))
        if len(cur_ori)==0:
            continue
        v_ori_prun.append(cur_ori.copy())

    # train data
    t_sents, t_tags = [],[]
    for i,ori in enumerate(t_ori_prun):
        cur_sent = ''.join(ori)
        pos = 0
        cur_tag = [0]*(len(cur_sent))
        if len(cur_sent)==0:
            #print(i,ori)
            continue
        for word in ori:
            if len(word)==1:
                # single word
                cur_tag[pos] = tag2id['S']
                pos += 1
            else:
                # more than one word
                cur_tag[pos] = tag2id['B']
                cur_tag[pos+len(word)-1] = tag2id['E']
                if len(word)>2:
                    cur_tag[pos+1:pos+len(word)-1] = [tag2id['M']]*(len(word)-2)
                pos = pos + len(word)
        
        t_sents.append(cur_sent)
        t_tags.append([0]+cur_tag+[0])


    # dev data
    v_sents, v_tags = [],[]
    for i,ori in enumerate(v_ori_prun):
        cur_sent = ''.join(ori)
        pos = 0
        cur_tag = [0]*(len(cur_sent))
        if len(cur_sent)==0:
            print(i,ori)
            continue
        #print(ori)
        for word in ori:
            if len(word)==1:
                # single word
                cur_tag[pos] = tag2id['S']
                pos += 1
            else:
                # more than one word
                cur_tag[pos] = tag2id['B']
                cur_tag[pos+len(word)-1] = tag2id['E']
                if len(word)>2:
                    cur_tag[pos+1:pos+len(word)-1] = [tag2id['M']]*(len(word)-2)
                pos = pos + len(word)
        
        v_sents.append(cur_sent)
        v_tags.append([0]+cur_tag+[0])


    # convert to token ids
    t_toks = []
    for i,sent in enumerate(t_sents):
        if i%100==0:
            print("\r%s%%"%(str(100*i/len(t_sents))[:10]), end='', flush=True)
        toks = list(sent)
        tok_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(toks) + [tokenizer.sep_token_id]
        t_toks.append(tok_ids)

    print("\n")
    v_toks = []
    for i,sent in enumerate(v_sents):
        if i%100==0:
            print("\r%s%%"%(str(100*i/len(v_sents))[:10]), end='', flush=True)
        toks = list(sent)
        tok_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(toks) + [tokenizer.sep_token_id]
        v_toks.append(tok_ids)


    dtset = MyDataset(t_toks, t_tags)
    train_loader = DataLoader(dataset=dtset, batch_size=train_batch, collate_fn=custom_collate, shuffle=True)
    dtset = MyDataset(v_toks, v_tags)
    dev_loader = DataLoader(dataset=dtset, batch_size=dev_batch, collate_fn=custom_collate, shuffle=False)

    if not extra_info:
        return train_loader, dev_loader
    else:
        return train_loader, dev_loader, extra_pos


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./data/pretrained')
    train_loader, dev_loader, extra_pos = load_dataset(tokenizer,4,4,extra_info=True)
    print(extra_pos)
    
    tmp = list(train_loader)
    a,b,c = tmp[10]
    print("\n")
    for aa,bb,cc in zip(a,b,c):
        d = tokenizer.convert_ids_to_tokens(aa.tolist())[:int(cc.sum())]
        #e = bb[:int(cc.sum())]
        e = bb
        print(len(d), len(e))
        for dd in d:
            print(dd+'\t', end='')
        print("\n")
        for bbb in e:
            print(id2tag[bbb.item()]+'\t', end='')
        print("\n")
