import time
from sklearn import metrics
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

from util import load_dataset
from model import BERT_LSTM

train_batch = 16
dev_batch = 16
LSTM_hid_dim = 512
tag2id = {'S':0, 'B':1, 'M':2, 'E':3}
id2tag = {0:'S', 1:'B', 2:'M', 3:'E'}

def show_scores(predictions, v_labels, valid_len):
    score_name = ['Micro precision', 'Macro precision', 'Micro recall', 'Macro recall',
              'Micro F1', 'Macro F1']
    scores = [0.]*6
    for preds, golds, v_len in zip(predictions, v_labels, valid_len):
        preds = preds[1:v_len+1]
        golds = golds[1:v_len+1]
        scores[0] += (metrics.precision_score(preds, golds, average='micro'))
        scores[1] += (metrics.precision_score(preds, golds, average='macro'))
        scores[2] += (metrics.recall_score(preds, golds, average='micro'))
        scores[3] += (metrics.recall_score(preds, golds, average='macro'))
        scores[4] += (metrics.f1_score(preds, golds, average='micro'))
        scores[5] += (metrics.f1_score(preds, golds, average='macro'))
    for i in range(len(scores)):
        scores[i] /= len(predictions)
    for na, sc in zip(score_name, scores):
        print(na, ': ', sc)
    return scores


def main(model_arch='bert_only'):
    tokenizer = BertTokenizer.from_pretrained('./data/pretrained')

    data_loader, dev_loader, extra_pos = load_dataset(tokenizer, train_batch, dev_batch, extra_info=True)
    

    if model_arch=='bert_only':
        output_model_file = './data/transformers_checkpoint/bert_only_checkpt_new.pth'
        model = BERT_ONLY(len(tag2id), load_pre=False)
    elif model_arch=='bert_lstm':
        output_model_file = './data/transformers_checkpoint/bert_lstm_checkpt_new.pth'
        model = BERT_LSTM(len(tag2id), hidden_dim=LSTM_hid_dim, load_pre=False, num_layers=2)
    elif model_arch=='bert_crf':
        output_model_file = './data/transformers_checkpoint/bert_crf_checkpt.pth'
        model = BERT_CRF(len(tag2id), tag2id, load_pre=False)
    model.load_state_dict(torch.load(output_model_file))
    
    eval_bert_lstm(model, data_loader, dev_loader, extra_pos)


def eval_bert_lstm(model, train_loader, dev_loader, extra_pos):
    
    print(len(train_loader), len(dev_loader))
    res_output_file = "./eval/test_pred_bert_lstm_3.txt"
    # log_output_file = "./result/log.dat"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # evaluate model
    model.eval()
    preds, golds, valid_len = [],[],[]
    ct = 0
    for sents, tags, masks in dev_loader:
        ct += 1
        if ct%20 == 19:
            print("\r%s%%"%(str(100*ct/len(dev_loader))[:8]), end='', flush=True)
        sents, tags, masks = sents.to(device), tags.to(device), masks.to(device)
        with torch.no_grad():
            logits = model(sents, masks)

            # predictions with regard to max score
            preds += torch.max(logits, 2)[1].tolist()
            # correct tags on dev set
            golds += tags.tolist()
            # since we must evaluate performance on raw sentence, get the length of raw sents (-2: <CLS> & <SEP>) 
            valid_len += [int(a.sum().item()-2) for a in masks]
    
    show_scores(preds, golds, valid_len)[-1]

    all_sents = []
    for sents,_,__ in dev_loader:
            all_sents += sents.tolist()

    tokenizer = BertTokenizer.from_pretrained('./data/pretrained')
    f = open(res_output_file, 'w', encoding='utf-8')
    for i, cur_preds, cur_len, cur_sent in zip(range(len(preds)), preds, valid_len, all_sents):
        if i not in extra_pos and i!=0:
            f.write("\n")
        elif i in extra_pos:
            f.write("  ")
        cur_full_sent = "".join(tokenizer.convert_ids_to_tokens(cur_sent[1:cur_len+1]))
        cur_seg = []
        cur_state = 0
        for p, lab in enumerate(cur_preds[1:cur_len+1]):
            if lab == tag2id['S']:
                if cur_state == 1:
                    # invalid output, directly save previous sub-sequence
                    cur_seg.append(cur_full_sent[p1:p])
                    cur_state = 0

                cur_seg.append(cur_full_sent[p:p+1])

            elif lab == tag2id['B']:
                if cur_state == 1:
                    cur_seg.append(cur_full_sent[p1:p])
                cur_state = 1
                p1 = p

            elif lab == tag2id['E']:
                if cur_state == 0:
                    p1 = p
                cur_state = 0
                cur_seg.append(cur_full_sent[p1:p+1])

            elif lab == tag2id['M']:
                if cur_state != 1:
                    cur_state = 1
                    p1 = p
        
        f.write("  ".join(cur_seg))

    f.close()
    model.train()


if __name__ == '__main__':
    main(model_arch='bert_lstm')