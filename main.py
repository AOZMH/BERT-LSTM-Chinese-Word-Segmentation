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


def show_scores(predictions, v_labels, valid_len, f):
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
        f.write(na+': '+str(sc)[:7]+'\n')
    return scores


def main(update_bert=False, load_chkpt=False, model_arch='bert_only'):
    tokenizer = BertTokenizer.from_pretrained('./data/pretrained')

    data_loader, dev_loader = load_dataset(tokenizer, train_batch, dev_batch)
    
    if update_bert:
        # update the embedding shape because vocab size may have changed
        bert_model = BertModel.from_pretrained('./data/transformers_pretrained')
        bert_model.resize_token_embeddings(len(tokenizer))
        bert_model.save_pretrained('./data/transformers_pretrained')
        bert_model.config.to_json_file('./data/transformers_pretrained/config.json')

    if not load_chkpt:
        if model_arch=='bert_lstm_crf':
            model = BERT_LSTM_CRF(LSTM_hid_dim, len(id2tag), tag2id, bert_route='./data/transformers_pretrained')
        elif model_arch=='bert_only':
            model = BERT_ONLY(len(tag2id), bert_route='./data/pretrained', load_pre=True)
        elif model_arch=='bert_lstm':
            model = BERT_LSTM(len(tag2id), hidden_dim=LSTM_hid_dim, bert_route='./data/pretrained', load_pre=True, num_layers=2)
        elif model_arch=='bert_crf':
            model = BERT_CRF(len(id2tag), tag2id, bert_route='./data/transformers_pretrained')
    else:
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
    
    train_bert_lstm(model, data_loader, dev_loader)


def train_bert_lstm(model, train_loader, dev_loader):
    
    print(len(train_loader), len(dev_loader))
    best_macro_f1 = 0.9798
    output_model_file = './data/transformers_checkpoint/bert_lstm_checkpt_new.pth'
    log_output_file = "./result/Bert_Lstm/log_"+time.ctime()[4:19]+".dat"
    # log_output_file = "./result/log.dat"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # optimizer defined after transfer to GPU
    optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.7)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()

    for epoch in range(3000):
        running_loss = 0.

        for batch_i, (sents, tags, masks) in enumerate(train_loader):
            
            sents, tags, masks = sents.to(device), tags.to(device), masks.to(device)

            optimizer.zero_grad()
            out_feats = model(sents, masks)
            loss = torch.tensor(0.).to(device)
            # Since lossf can only calculate 2-dim output & 1-dim tag, recusively cal loss
            for b_feats, b_tags in zip(out_feats, tags):
                loss += lossf(b_feats, b_tags)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_i % 100 == 99:  # print every 100 mini-batches
                f = open(log_output_file, 'a', encoding='utf-8')
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, batch_i + 1, running_loss / 10))
                f.write('[%d, %5d] loss: %.5f\n' %
                      (epoch + 1, batch_i + 1, running_loss / 10))
                running_loss = 0.0
                f.close()
        
        # evaluate every epoch
        if epoch%1 == 0:
            model.eval()
            preds, golds, valid_len = [],[],[]
            for sents, tags, masks in dev_loader:
                sents, tags, masks = sents.to(device), tags.to(device), masks.to(device)
                with torch.no_grad():
                    logits = model(sents, masks)

                    # predictions with regard to max score
                    preds += torch.max(logits, 2)[1].tolist()
                    # correct tags on dev set
                    golds += tags.tolist()
                    # since we must evaluate performance on raw sentence, get the length of raw sents (-2: <CLS> & <SEP>) 
                    valid_len += [int(a.sum().item()-2) for a in masks]
                
            f = open(log_output_file, 'a', encoding='utf-8')
            f.write("Epoch: "+str(epoch)+'\n')
            cur_macro_f1 = show_scores(preds, golds, valid_len, f)[-1]
            print("Current best f1-macro: %s"%(str(max(cur_macro_f1, best_macro_f1))[:8]))
            f.write("Current best f1-macro: %s\n"%(str(max(cur_macro_f1, best_macro_f1))[:8]))
            f.close()

            if cur_macro_f1 > best_macro_f1:
                best_macro_f1 = cur_macro_f1
                # need to show results, thus get all input raw sentences
                all_sents = []
                for sents,_,__ in dev_loader:
                    all_sents += sents.tolist()
                # output to result file
                f = open("./result/Bert_Lstm/result_epoch"+str(cur_macro_f1)[:6]+".dat", "a", encoding='utf-8')
                # reload tokenizer to recover raw sentence
                tokenizer = BertTokenizer.from_pretrained('./data/pretrained')

                for cur_preds, cur_golds, cur_len, cur_sent in zip(preds, golds, valid_len, all_sents):
                    f.write('Ques:\t' + "\t".join(tokenizer.convert_ids_to_tokens(cur_sent[1:cur_len+1])) +'\nPred:\t' +
                            "\t".join([id2tag[temp] for temp in cur_preds[1:cur_len+1]]) +'\nGold:\t'+
                            "\t".join([id2tag[temp] for temp in cur_golds[1:cur_len+1]])+'\n')
                f.close()

                print('Saving model...')
                torch.save(model.state_dict(), output_model_file)
                #model.bert_encoder.config.to_json_file(output_config_file)


            model.train()

if __name__ == '__main__':
    main(update_bert=False, load_chkpt=True, model_arch='bert_lstm')