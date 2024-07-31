# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
import pandas as pd
import h5py
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

# %%
class Model(nn.Module):
    def __init__(self, dim, hidden_dim=256, num_classes=2):
        super(Model, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(),    
            nn.Linear(hidden_dim, num_classes), #hidden_ 
        )
        
    def forward(self, x):
        return self.classifier(x)
    
    
# %%
""" Dataset: """
class FeatDataset(Dataset):
    def __init__(self,feat_list,df,target_label,target_dict):
        self.df = df
        self.feat_list = feat_list
        self.target_label = target_label
        self.target_dict = target_dict
        #self.nr_feats = nr_feats
        
    def __len__(self):
        return len(self.feat_list)
    def get_targets(self):
        #if self.tcga:
        return [self.target_dict[self.df[self.df.PATIENT==feat_path.split("/")[-1].split(".h5")[0].split(".")[0][:12]][self.target_label].values[0]]
                    for feat_path in self.feat_list]
    def get_nr_pos(self):
        if self.tcga:
            targets = np.array([self.target_dict[self.df[self.df.PATIENT==feat_path.split("/")[-1].split(".h5")[0].split(".")[0][:12]][self.target_label].values[0]]
                    for feat_path in self.feat_list])
            return len(targets[targets==1])
    def get_nr_neg(self):
        if self.tcga:
            targets = np.array([self.target_dict[self.df[self.df.PATIENT==feat_path.split("/")[-1].split(".h5")[0].split(".")[0][:12]][self.target_label].values[0]]
                    for feat_path in self.feat_list])
            return len(targets[targets==0])
    def get_patient_ids(self):
        #if self.tcga:
        return [self.df[self.df.PATIENT==feat_path.split("/")[-1].split(".h5")[0].split(".")[0][:12]].PATIENT.values[0]
                for feat_path in self.feat_list]
        
    def __getitem__(self, index):
        feat_path = self.feat_list[index]
        #if self.tcga:
        pat = feat_path.split("/")[-1].split(".h5")[0].split(".")[0][:12]
        try:
            target = self.target_dict[self.df[self.df.PATIENT==pat][self.target_label].values[0]]
        except Exception as exc:
            print(f"Exception: {exc}")
            target = 0
        try:
            feats = torch.from_numpy(h5py.File(feat_path)["feats"][:])
        except Exception:
            print(f"Problem with {feat_path}!")
            feats = torch.from_numpy(h5py.File(feat_path)["features"][:])
        #if len(feats)<self.nr_feats:
        #    feats = torch.cat((feats,torch.zeros(self.nr_feats - feats.shape[0],feats.shape[1])))
        return feats, target, pat
    
# %%
def train(target_label="isMSIH"):
    
    target_dict = {"nonMSIH":0,"MSIH":1}
    
    model = Model(dim=512).cuda()
    
    df = pd.read_excel(df_path)
    print(df)
    print(f"{len(df)=}")
    df = df.dropna(axis=0,subset=["isMSIH"])
    print(f"{len(df)=}")
    feat_files = np.random.permutation([os.path.join(feat_path,f) for f in os.listdir(feat_path) if f.split(".h5")[0] in df.PATIENT.values]).tolist()
    
    test_cases = int(np.ceil(0.2*len(feat_files)))
    val_cases = int(np.floor(0.2*len(feat_files)))
    
    test_files = feat_files[:test_cases]
    val_files = feat_files[test_cases:val_cases+test_cases]
    tr_files = feat_files[val_cases+test_cases:]
    
    tr_ds = FeatDataset(tr_files,df,target_label,target_dict)
    val_ds = FeatDataset(val_files,df,target_label,target_dict)
    test_ds = FeatDataset(test_files,df,target_label,target_dict)
    
    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=global_cfg["num_workers"])
    loader_dict = {"train":train_loader, "val":val_loader, "test":test_loader} #,"total":data_loader
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    positive_weights = compute_class_weight('balanced', classes=sorted(list(loader_dict["train"].dataset.target_dict.values())), 
                                        y=loader_dict["train"].dataset.get_targets())
    positive_weights = torch.tensor(positive_weights, dtype=torch.float).cuda()
    print(f"{positive_weights=}")
    criterion = nn.CrossEntropyLoss(weight=positive_weights)

    best_val_loss = float('inf')  # Initialize with a large value
    best_model_state_dict = None

    breakpoint = False

    pbar = tqdm(total=epochs, desc='Training Progress', unit='epoch')
    
    for e in range(epochs):
        
        model.train()
        
        if breakpoint:
            break
            
        total_train_loss = 0.0
        total_train_correct = 0

        total_val_loss = 0.0
        total_val_correct = 0
        
        for mode in ["train","val"]:
            for feats, targets, _ in tqdm(loader_dict[mode],leave=False):
                if torch.cuda.is_available():
                    feats = feats.cuda()
                    targets = targets.cuda().to(torch.long)
                    #targets_adv = targets_adv.cuda().to(torch.long)
                #model.manipulator.eval()
                #feats = model.manipulate(feats,comb_cfg["man_range_low"],comb_cfg["man_range_high"])
                if mode == "val":
                    model.eval()
                    with torch.no_grad():
                        logits = model(feats)
                        loss = criterion(logits, targets)
                        #loss_dyn = criterion_adv(logits_dyn, targets_adv)
                        #loss_stat = criterion_adv(logits_stat, targets_adv)
                        total_val_loss += loss.item()
                        _, predicted_labels = torch.max(logits, dim=1)
                        total_val_correct += (predicted_labels == targets).sum().item()
                    
                else:
                    logits = model(feats)
                    loss = criterion(logits, targets)
                    #loss_dyn = criterion_adv(logits_dyn, targets_adv)
                    #loss_stat = criterion_adv(logits_stat, targets_adv)
                    optimizer.zero_grad() 
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                    _, predicted_labels = torch.max(logits, dim=1)
                    total_train_correct += (predicted_labels == targets).sum().item()

            if mode=="val":
                # Checkpointing: Save the model with the lowest validation loss
                avg_tr_loss = total_train_loss / len(train_loader)
                avg_tr_acc = total_train_correct / len(tr_ds)
                
                avg_val_loss = total_val_loss / len(val_loader)
                avg_val_acc = total_val_correct / len(val_ds)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state_dict = model.state_dict()
                    stop_count = 0
                    
                else:
                    stop_count += 1
                    if stop_count >= es:
                        tqdm.write("Early stopping triggered!")
                        breakpoint = True
                        break
                tqdm.write(f'Epoch: TRAIN: {e+1}, loss: {avg_tr_loss:.4f}, acc: {avg_tr_acc:.4f}, val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
        pbar.set_postfix(
        loss=avg_tr_loss,
        acc=avg_tr_acc,
        val_loss=avg_val_loss,
        val_acc=avg_val_acc,
        )
        pbar.update(1)
    pbar.close()
    #torch.save(best_model_state_dict,model_name)
    model.load_state_dict(best_model_state_dict)
    
    return model, loader_dict


def test(model,out_dir="figs",target_label="isMSIH",figs=False):
    # Evaluate the model on the test set
    model.eval()
    total_test_correct = 0
    all_predicted_probs = []
    all_targets = []
    all_pred_labels = []

    pred_dict = {"PATIENT":loader_dict["test"].dataset.get_patient_ids(),"preds":[], target_label:[]}

    test_loss = 0
    
    positive_weights = compute_class_weight('balanced', classes=sorted(list(loader_dict["train"].dataset.target_dict.values())), y=loader_dict["train"].dataset.get_targets())
    print(f"{positive_weights=}")
    #positive_weights = torch.tensor(positive_weights[1]/positive_weights[0], dtype=torch.float).cuda()
    positive_weights = torch.tensor(positive_weights, dtype=torch.float).cuda()

    criterion = nn.CrossEntropyLoss(weight=positive_weights)
       
    aurocs = []
    
    for feats, targets, _ in tqdm(loader_dict["test"],leave=False):
        if torch.cuda.is_available():
            feats = feats.cuda()
            #targets = nn.functional.one_hot(targets,num_classes=num_classes).cuda().to(torch.int64)
            targets = targets.cuda()

        with torch.no_grad():
            logits = model(feats)
            loss = criterion(logits,targets)
            test_loss += loss.item()
        
            _, predicted_labels = torch.max(logits, dim=1)
            total_test_correct += (predicted_labels == targets).sum().item()

            all_pred_labels.append(predicted_labels.cpu().numpy())
            predicted_probs = nn.functional.softmax(logits, dim=1)
            pred_dict["preds"].extend(predicted_probs[:,1].cpu().numpy().flatten().tolist())
            pred_dict[target_label].extend(targets.cpu().numpy().flatten().tolist())
            #pred_dict["PATIENT"].extend(pats.cpu().numpy().flatten().tolist())
            all_predicted_probs.append(predicted_probs[:, 1].cpu().numpy())  
            all_targets.append(targets.cpu().numpy())

    # Calculate test accuracy
    test_accuracy = total_test_correct / len(loader_dict["test"].dataset)

    # Flatten the predicted probabilities and targets
    all_predicted_probs = np.concatenate(all_predicted_probs)
    all_targets = np.concatenate(all_targets)
    all_pred_labels = np.concatenate(all_pred_labels)

    test_loss_avg = test_loss / len(loader_dict["test"])
    assert len(np.unique([len(pred_dict[k]) for k in pred_dict.keys()]))==1, f"the lengths of the lists are different: {[len(pred_dict[k]) for k in pred_dict.keys()]}"

    test_df = pd.DataFrame(pred_dict)
    
    # if not exists(f"{out_dir}/preds"):
    #     Path(f"{out_dir}/preds").mkdir(parents=True, exist_ok=True)
    
    # test_df.to_csv(f"{out_dir}/preds/PMA_mil_preds_test-{target_label}-{nr}.csv",index=False)
    test_auroc = roc_auc_score(all_targets, all_predicted_probs)
    # Calculate AUROC
    if figs:
        
        aurocs.append(test_auroc)

        fpr, tpr, _ = roc_curve(all_targets, all_predicted_probs)

        precision, recall, _ = precision_recall_curve(all_targets, all_predicted_probs)
        
        fig_roc = plt.figure(figsize=(26,15))
        ax_roc = fig_roc.add_subplot(111)


        fig_prc = plt.figure(figsize=(26,15))
        ax_prc = fig_prc.add_subplot(111)
        fig_prc.suptitle("Precision-Recall Curve",fontsize=24) 

        ax_roc.plot(fpr, tpr, label=f"AUC={test_auroc:.3f}")

        ax_prc.plot(recall,precision,label=f"PRC")

        tqdm.write(f"Test loss: {test_loss_avg:.4f}, Test Acc: {test_accuracy*100:.2f}, AUROC: {test_auroc:.4f}")

        fig_roc.legend(loc="lower left",fontsize=16)
        fig_prc.legend(loc="lower left",fontsize=16)
        ax_roc.plot([0, 1], [0, 1], 'r--')
        ax_prc.tick_params(axis='both', which='both', labelsize=22)
        ax_roc.tick_params(axis='both', which='both', labelsize=22)


        ax_prc.set_xlabel('Recall',fontsize=24)
        ax_prc.set_ylabel('Precision',fontsize=24)
        ax_roc.set_xlabel('False Positive Rate',fontsize=24)
        ax_roc.set_ylabel('True Positive Rate',fontsize=24)
        ax_roc.set_aspect("equal")
        fig_roc.suptitle(f'ROC AVG AUROC: {np.mean(aurocs):.3f}$\pm${np.std(aurocs):.3f}',fontsize=24)

        fig_roc.savefig(f"{out_dir}/ROC-mil-{target_label}.pdf",dpi=300)
        fig_prc.savefig(f"{out_dir}/PRC-mil-{target_label}.pdf",dpi=300)
    
    else:
        tqdm.write(f"Test loss: {test_loss_avg:.4f}, Test Acc: {test_accuracy*100:.2f}")
    print(f"AUROC: {test_auroc:.4f}")
    print(f"F1 score (weighted): {f1_score(all_targets, all_pred_labels,average='weighted'):.5f}")
    print(f"F1 score (micro): {f1_score(all_targets, all_pred_labels,average='micro'):.5f}")
    print(f"F1 score (macro): {f1_score(all_targets, all_pred_labels,average='macro'):.5f}")
# %%
# params
lr = 1e-4
epochs = 500
es = 50
batch_size = 32
num_workers = 10

#kwargs = {"lr":lr,"epochs":epochs,"es":es,"batch_size":batch_size,"num_workers":num_workers}

df_path = "/mnt/bulk-neptune/timlenz/tumpe/data/targets/TCGA-CRC-DX_CLINI.xlsx"
feat_path = "/mnt/bulk-neptune/timlenz/tumpe/data/features/caduceus-mamba-2-1024-e10"

model, loader_dict = train()
test(model,loader_dict)

