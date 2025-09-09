import os, torch, pickle
from torch.utils.data import DataLoader
from base_model import BaseModel


def train_one(trainset, testset, device, epochs=20, batch_size=128, lr=0.001, wd=5e-4):
    mdl = BaseModel("resnet18", device=device, num_cls=10, input_dim=3,
                    optimizer="adam", lr=lr, weight_decay=wd, epochs=epochs)
    tr = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    te = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    best = (-1, None)
    best_state = None
    for ep in range(epochs):
        mdl.train(tr, log_pref=f"Train ep{ep}")
        acc, loss = mdl.test(te, log_pref=f"Val  ep{ep}")
        if acc > best[0]:
            best = (acc, ep)
            best_state = {'epoch': ep+1, 'acc': acc, 'loss': loss, 'state': mdl.model.state_dict()}
    torch.save(best_state, "best_tmp.pth")
    return mdl, "best_tmp.pth"


def main(dataset="cifar10", model="resnet18", device="cuda:0", shadow_num=5):
    save_root = f"result/{dataset}_{model}"
    os.makedirs(save_root, exist_ok=True)
    with open(f"{save_root}/data_prepare.pkl", "rb") as f:
        victim_train_list, victim_train_dataset, victim_dev_dataset, victim_test_dataset, attack_split_list, shadow_train_list = pickle.load(f)

    victim_model_dir = f"{save_root}/victim_model"
    os.makedirs(victim_model_dir, exist_ok=True)
    _, tmp = train_one(victim_train_dataset, victim_dev_dataset, device)
    os.replace(tmp, f"{victim_model_dir}/best.pth")
    print("saved:", f"{victim_model_dir}/best.pth")

    for i, (tr, dv, te) in enumerate(attack_split_list):
        d = f"{save_root}/shadow_model_{i}"
        os.makedirs(d, exist_ok=True)
        _, tmp = train_one(tr, dv, device)
        os.replace(tmp, f"{d}/best.pth")
        print("saved:", f"{d}/best.pth")


if __name__ == "__main__":
    main()

