import os, torch, torch.nn.utils.prune as prune, torch.nn as nn
from torch.utils.data import DataLoader
from base_model import BaseModel
import pickle


def load_base(model_name, num_cls, device, ckpt_path, input_dim=3):
    mdl = BaseModel(model_name, device=device, num_cls=num_cls, input_dim=input_dim,
                    optimizer="adam", lr=0.001, weight_decay=5e-4, epochs=5)
    state = torch.load(ckpt_path, map_location=device)
    mdl.model.load_state_dict(state['state'])
    return mdl


def global_l1_prune(model, amount=0.6):
    parameters_to_prune = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((m, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for m, _ in parameters_to_prune:
        prune.remove(m, 'weight')
    return model


def finetune(mdl, trainset, testset, device, epochs=5, batch_size=128):
    tr = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    te = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    best = -1
    best_state = None
    for ep in range(epochs):
        mdl.train(tr, log_pref=f"[FT] ep{ep}")
        acc, loss = mdl.test(te, log_pref=f"[FT] val{ep}")
        if acc > best:
            best, best_state = acc, mdl.model.state_dict()
    return best, best_state


def prune_one(save_dir, src_ckpt, trainset, testset, device, prune_sparsity=0.6):
    os.makedirs(save_dir, exist_ok=True)
    mdl = load_base("resnet18", 10, device, src_ckpt)
    mdl.model = global_l1_prune(mdl.model, amount=prune_sparsity)
    _, state = finetune(mdl, trainset, testset, device)
    torch.save(state, f"{save_dir}/best.pth")
    print("saved pruned:", f"{save_dir}/best.pth")


def main(dataset="cifar10", model="resnet18", device="cuda:0", prune_sparsity=0.6, shadow_num=5):
    save_root = f"result/{dataset}_{model}"
    with open(f"{save_root}/data_prepare.pkl", "rb") as f:
        victim_train_list, v_tr, v_dev, v_te, attack_split_list, shadow_train_list = pickle.load(f)

    victim_src = f"{save_root}/victim_model/best.pth"
    victim_dst_dir = f"{save_root}/l1unstructure_{prune_sparsity}_model"
    prune_one(victim_dst_dir, victim_src, v_tr, v_dev, device, prune_sparsity)

    for i, (tr, dv, te) in enumerate(attack_split_list):
        src = f"{save_root}/shadow_model_{i}/best.pth"
        dst = f"{save_root}/shadow_l1unstructure_{prune_sparsity}_model_{i}"
        prune_one(dst, src, tr, dv, device, prune_sparsity)


if __name__ == "__main__":
    main()

