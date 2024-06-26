import torch
import time
from utils.utils import parse_args, setup_seed, random_sampler
from utils.models import BaseMF
from utils.dataloader import get_dataloader, get_data
from utils.evaluate import calculate_noranking_fairness, New_Eval, deg, display_all_results, display_noranking_results, f1
from utils.optimtools import get_preference_vectors, batch_group_probs, get_d_paretomtl_init, get_d_paretomtl_recloss,get_d_paretomtl_infoentropy
from torch.autograd import Variable
import math,heapq
import torch.nn as nn
import numpy as np
import pickle
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class EarlyStopping:    # different from EarlyStopping from utils.utils!
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = 99
        self.early_stop = False
        self.best_state = None
        self.inzone = True

    def __call__(self, score, model, inzone):
        if self.inzone is False and inzone is True:
            self.save_checkpoint(score, model)
            self.inzone = True
        elif self.inzone == inzone and score <= self.best_score:
            self.save_checkpoint(score, model)
        else:   # (self.inzone is True and inzone is False) or (self.inzone == inzone and score < self.best_score)
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
        
    def save_checkpoint(self, score, model):
        self.best_state = {key: value.cpu() for key, value in model.state_dict().items()}                
        self.best_score = score
        self.counter = 0


def found_initial_solution(args, rsd, rsdn):
    degree = deg([rsd, rsdn])
    mindeg, maxdeg = 90 * args.pref_idx / args.npref, 90 * (args.pref_idx+1) / args.npref
    if degree > mindeg and degree < maxdeg:
        return True
    return False

#计算信息熵

def calculate_Info_Entropy(item_groups, users, rec, topic_num, K=20):
    num_topics = 6  
    total_entropy = 0    
    for u in users:
        heap = rec[u].to('cpu')
        # 获取前K个推荐项
        topK = torch.topk(heap, K).indices
        map_topic2num = torch.zeros(num_topics, dtype=torch.float32)      
        # 计算每个组中有多少项
        for k in topK:
            map_topic2num[item_groups[k].item()] += 1       
        # 包括正样本（所有交互过的项）
        map_topic2num += torch.tensor(topic_num[u.item()], dtype=torch.float32).to('cpu') 
        total_count = torch.sum(map_topic2num)
        prob_distribution = map_topic2num / total_count          
        # 计算信息熵
        entropy = -torch.sum(prob_distribution * torch.log(prob_distribution + 1e-9)) / math.log(num_topics)
        total_entropy += entropy.item()    
    return total_entropy / len(users)


def train_model(item_groups, user_neighbors):
    ref_vec = torch.tensor(get_preference_vectors(npref)).to(device).float()    # circle_points([1], [npref])[0]
    print('Preference Vector ({}/{}):'.format(pref_idx + 1, npref))
    print(ref_vec[pref_idx].cpu().numpy(), 90*(0.5+pref_idx)/npref)
    optimizer_initial = torch.optim.Adam(model.parameters(), lr=args.initsol_lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    earlystopper = EarlyStopping(patience=args.patience)    # default: 5
    Rec = model.predict().detach().cpu().numpy()
    prob_list, nprob_list = calculate_noranking_fairness(args, Rec, val_dict, train_list, item_groups, user_neighbors)
    _, _ = display_noranking_results(args, prob_list, nprob_list)
    
    print('----- Finding the initial solution -----')
    for step in range(args.initsol_epoch):
        torch.cuda.empty_cache()
        print('********** Step {} **********'.format(step+1))
        item_groups, user_neighbors = torch.LongTensor(item_groups).to(device), torch.LongTensor(user_neighbors).to(device)
        model.train()
        for _, (users, pos_items) in enumerate(train_loader):
            grads, losses_vec = {0:[],1:[]}, []
            neg_items = random_sampler(users.numpy(), pos_items.numpy(), train_list, args)
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            pos_scores, neg_scores = model.calculate_pos_neg_scores(users, pos_items, neg_items)
            speo_pt, nspneo_pt = batch_group_probs(args, users, item_groups, user_neighbors, pos_items, neg_items, pos_scores, neg_scores)
            rsd_speo, rsd_nspneo = torch.std(speo_pt) / torch.mean(speo_pt), torch.std(nspneo_pt) / torch.mean(nspneo_pt)

            optimizer_initial.zero_grad()   # SP/EO
            rsd_speo.backward(retain_graph=True)
            for param in model.parameters():
                if param.grad is not None:
                    grads[0].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            
            optimizer_initial.zero_grad()   # NSP/NEO
            rsd_nspneo.backward(retain_graph=True)
            for param in model.parameters():
                if param.grad is not None:
                    grads[1].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            
            grads = torch.stack([torch.cat(grads[i]) for i in range(2)])
            losses_vec = torch.stack([rsd_speo.data, rsd_nspneo.data])    # calculate the weights
            _, weight_vec = get_d_paretomtl_init(grads, losses_vec, ref_vec, pref_idx)
            optimizer_initial.zero_grad()
            loss_total = weight_vec[0] * rsd_speo + weight_vec[1] * rsd_nspneo
            loss_total.backward()
            optimizer_initial.step()
        
        model.eval()
        Rec = model.predict().detach().cpu().numpy()
        item_groups, user_neighbors = item_groups.cpu().numpy().tolist(), user_neighbors.cpu().numpy().tolist()
        prob_list, nprob_list = calculate_noranking_fairness(args, Rec, val_dict, train_list, item_groups, user_neighbors)
        rsd, rsdn = display_noranking_results(args, prob_list, nprob_list)
        if found_initial_solution(args, rsd, rsdn) is True:
            print('^^^^^ Initial Solution Is Found!!! ^^^^^')
            break
    if found_initial_solution(args, rsd, rsdn) is False:
        print('^^^^^ Warning: Initial Solution Not Found!!! ^^^^^')
        earlystopper.inzone = False
        
        
    epoch_info_loss=[]
    epoch_info_entropy=[]
    print('----- Running our method -----')
    for epoch in range(1, args.max_epoch+1):
        torch.cuda.empty_cache()
        print('********** Epoch {} **********'.format(epoch))
        epoch_start_time = time.time()
        total_neg_sample_time, total_train_inference_time, total_pareto_time, total_bprloss, total_regloss = 0, 0, 0, 0, 0
        item_groups, user_neighbors = torch.LongTensor(item_groups).to(device), torch.LongTensor(user_neighbors).to(device)
        model.train()
        for _, (users, pos_items) in enumerate(train_loader):
            grads, losses_vec = {0:[],1:[],2:[],3:[]}, []
            sample_start_time = time.time()
            neg_items = random_sampler(users.numpy(), pos_items.numpy(), train_list, args)
            sample_end_time = time.time()

            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            pos_scores, neg_scores = model.calculate_pos_neg_scores(users, pos_items, neg_items)
            bprloss, regloss = model.calculate_loss(pos_scores, neg_scores)
            recloss = bprloss + args.reg*regloss
            total_bprloss += bprloss.detach().cpu().item()
            total_regloss += args.reg*regloss.detach().cpu().item()
            
            info_entropy=calculate_Info_Entropy(item_groups.cpu(), users.cpu(), model.predict().detach().cpu(), topic_num)
            epoch_info_entropy.append(info_entropy)
            info_loss=torch.nn.functional.binary_cross_entropy(torch.sigmoid(model.predict()), target)
            epoch_info_loss.append(info_loss)

            pareto_start_time = time.time()
            speo_pt, nspneo_pt = batch_group_probs(args, users, item_groups, user_neighbors, pos_items, neg_items, pos_scores, neg_scores)
            rsd_speo, rsd_nspneo = torch.std(speo_pt) / torch.mean(speo_pt), torch.std(nspneo_pt) / torch.mean(nspneo_pt)
            for i in range(4):
                optimizer.zero_grad()
                if i==0:
                    rsd_speo.backward(retain_graph=True)
                elif i==1:
                    rsd_nspneo.backward(retain_graph=True)
                elif i==2:
                    recloss.backward(retain_graph=True)
                else:
                    info_loss.backward(retain_graph=True)
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads = torch.stack([torch.cat(grads[i]) for i in range(4)])
            losses_vec = torch.stack([rsd_speo.data, rsd_nspneo.data])    # calculate the weights
            
            weight_vec = get_d_paretomtl_infoentropy(grads, losses_vec, ref_vec, pref_idx, recloss, args.thre,info_loss,args.thre_ie)
            #normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            #weight_vec = weight_vec * normalize_coeff
            pareto_end_time = time.time()

            if len(weight_vec) == 2:
                normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
                weight_vec = weight_vec * normalize_coeff
                loss_total = weight_vec[0] * rsd_speo + weight_vec[1] * rsd_nspneo
                
                #考虑rec loss不考虑信息熵
            elif len(weight_vec) == 3:
                normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
                weight_vec = weight_vec * normalize_coeff
                loss_total = weight_vec[0] * rsd_speo + weight_vec[1] * rsd_nspneo + weight_vec[2] * recloss
                
                #此时不考虑rec loss考虑信息熵
            else:
                weight_vec= weight_vec[:-1]
                normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
                weight_vec = weight_vec * normalize_coeff
                loss_total = weight_vec[0] * rsd_speo + weight_vec[1] * rsd_nspneo + 2 * weight_vec[2] * info_loss
            
            loss_total.backward()
            optimizer.step()

            total_train_inference_time += (pareto_start_time + time.time() - sample_end_time - pareto_end_time)
            total_pareto_time += (pareto_end_time - pareto_start_time)
            total_neg_sample_time += (sample_end_time - sample_start_time)
        print("Information Entropy:",sum(epoch_info_entropy)/len(epoch_info_entropy))
        print("Information Entropy Loss:",sum(epoch_info_loss)/len(epoch_info_loss))
        avgbpr, avgreg, avgloss = total_bprloss / len(train_loader), total_regloss / len(train_loader), (total_regloss+total_bprloss) / len(train_loader)
        print('Time:{:.2f}s = sample_{:.2f}s + trainfer_{:.2f} + pareto_{:.2f}s\tAvgRecLoss:{:.4f} = bpr_{:.4f} + reg_{:.4f}'.format(
            time.time()-epoch_start_time, total_neg_sample_time, total_train_inference_time, total_pareto_time, avgloss, avgbpr, avgreg
        ))
        Rec = model.predict().detach().cpu().numpy()
        item_groups, user_neighbors = item_groups.cpu().numpy().tolist(), user_neighbors.cpu().numpy().tolist()
        prob_list, nprob_list = calculate_noranking_fairness(args, Rec, val_dict, train_list, item_groups, user_neighbors)
        rsd, rsdn = display_noranking_results(args, prob_list, nprob_list)
        f1value, inzone = f1(rsd,rsdn), found_initial_solution(args, rsd, rsdn)
        if earlystopper(f1value, model, inzone) is True:
            break
    print('Loading {}th epoch'.format(min(epoch-args.patience, args.max_epoch)))
    model.load_state_dict(earlystopper.best_state)
    model.eval()
    Rec = model.predict().detach().cpu().numpy()
    print('********** Validating **********')
    _, _, ndcg, sp_list, nsp_list, eo_list, neo_list = New_Eval(args, Rec, val_dict, train_list, item_groups, user_neighbors, K=args.K)
    display_all_results(ndcg, sp_list, nsp_list, eo_list, neo_list)
    print('********** Testing **********')
    _, _, ndcg, sp_list, nsp_list, eo_list, neo_list = New_Eval(args, Rec, test_dict, trainval_list, item_groups, user_neighbors, K=args.K)
    display_all_results(ndcg, sp_list, nsp_list, eo_list, neo_list)


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device, args.reg = device, args.reg / 2
    n_tasks, npref, pref_idx = 2, args.npref, args.pref_idx
    assert args.pref_idx < args.npref
    assert args.mode in {'sp','eo'}
    assert args.dataset in {'KuaiRec','Epinions'}
    train_dict, val_dict, test_dict, num_user, num_item, num_train, _, _, item_groups, user_neighbors, _, _ = get_data(args.dataset)
    args.num_user, args.num_item, args.num_train, args.num_group = num_user, num_item, num_train, max(item_groups)+1
    train_loader, train_list, trainval_list = get_dataloader(args, train_dict, val_dict)
    

  
    input_tensor = torch.zeros(num_user, num_item)
    target = torch.sigmoid(input_tensor).to(device)
    target.requires_grad = True
    
    #计算训练集各用户交互过item的topic的个数
    topic_num={}
    for u in train_dict.keys() :
        user_topic_num = torch.zeros(6, dtype=torch.int64)
        for i in train_dict[u]:
            user_topic_num[item_groups[i]] += 1
        topic_num[u]=user_topic_num
        
    model = BaseMF(args).to(device)
    model.load_state_dict(torch.load('./data/{}/best_model.pth.tar'.format(args.dataset)))
    train_model(item_groups, user_neighbors)
