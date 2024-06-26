import torch
import time
from utils.utils import parse_args, setup_seed, random_sampler
from utils.models import BaseMF
from utils.dataloader import get_dataloader, get_data
from utils.evaluate import New_Eval, display_all_results, evaluate
import math
import heapq

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_state = None

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        return self.early_stop
        
    def save_checkpoint(self, val_score, model):
        self.best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        self.best_val_score = val_score

def calculate_Info_Entrophy(item_groups, users, pos_items, rec, K=20):
    num_topics = 6  
    total_entropy = 0
    
    for u,pos_item in zip(users,pos_items):
        heap = rec[u]
        #前20个推荐的item
        topK = torch.topk(torch.from_numpy(heap), K).indices 
        map_topic2num = torch.zeros(num_topics, dtype=torch.float32)
        
        # 每个group各有多少个item
        for k in topK:
            map_topic2num[item_groups[k]] += 1
            
        #把正样本也计入(每次训练值考虑一个正样本)
        map_topic2num[item_groups[pos_item]] += 1
        
        total_count = torch.sum(map_topic2num)
        prob_distribution = map_topic2num / total_count    
        
        # 计算信息熵
        entropy = -torch.sum(prob_distribution * torch.log(prob_distribution + 1e-9)) / math.log(6)
        total_entropy += entropy.item()
    
    return total_entropy / len(users)

def train_model():
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, train_list, trainval_list = get_dataloader(args, train_dict, val_dict)
    earlystopper = EarlyStopping(patience=args.patience)
    All_entro=[]
    for epoch in range(1, args.max_epoch+1):
        print('********** Epoch {} **********'.format(epoch))
        total_bprloss, total_regloss = 0, 0
        total_neg_sample_time, total_train_time = 0, 0
        # model.train()
        total_entro=[]
        for _, (users, pos_items) in enumerate(train_loader):
            ################################################
            info_e=-calculate_Info_Entrophy(item_groups,users,pos_items,model.predict().cpu().numpy())
            total_entro.append(info_e)
            print(info_e)
            #################################################
            optimizer.zero_grad()
            start_time = time.time()
            neg_items = random_sampler(users.numpy(), pos_items.numpy(), train_list, args)
            neg_sample_time = time.time()
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            pos_scores, neg_scores = model.calculate_pos_neg_scores(users, pos_items, neg_items)
            bprloss, regloss = model.calculate_loss(pos_scores, neg_scores)
            
            loss = bprloss + args.reg*regloss
            loss.backward()
            optimizer.step()
            train_time = time.time()
            total_bprloss += bprloss.detach().cpu().item()
            total_regloss += args.reg*regloss.detach().cpu().item()
            total_neg_sample_time += (neg_sample_time - start_time)
            total_train_time += (train_time - neg_sample_time)
        #################################################    
        All_entro.append(sum(total_entro)/len(total_entro))
        #################################################
        
        avgbpr, avgreg = total_bprloss / len(train_loader), total_regloss / len(train_loader)
        print('Time:{:.2f}s = sample_{:.2f}s + train_{:.2f}s\tavgbpr={:.4f}\tavgreg={:.4f}'.format(
            total_train_time + total_neg_sample_time, total_neg_sample_time, total_train_time, avgbpr, avgreg))
        start_time = time.time()
        # model.eval()
        Rec = model.predict().cpu().numpy()
        result = evaluate(Rec, val_dict, train_list)
        if earlystopper(result[1,1], model) is True:
            break
   #################################################        
    print(All_entro)
    print(torch.mean(All_entro))
    #################################################
    
    print('Loading {}th epoch'.format(min(epoch-args.patience, args.max_epoch)))
    model.load_state_dict(earlystopper.best_state)
    #torch.save(model.state_dict(), './data/{}/best_model.pth.tar'.format(args.dataset))
    # model.eval()
    Rec = model.predict().cpu().numpy()
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
    args.device = device
    args.reg = args.reg / 2
    train_dict, val_dict, test_dict, num_user, num_item, num_train, num_val, num_test, \
     item_groups, user_neighbors, group_item_dict, group_num_dict = get_data(args.dataset)
    args.num_user, args.num_item, args.num_train, args.num_group = num_user, num_item, num_train, max(item_groups)+1
    model = BaseMF(args).to(device)
    train_model()
