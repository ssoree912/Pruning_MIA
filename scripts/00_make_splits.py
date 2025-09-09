import os, pickle, random, numpy as np, torch
from torch.utils.data import Subset
from datasets import get_dataset


def main(dataset_name="cifar10", model_name="resnet18", seed=7, shadow_num=5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    trainset = get_dataset(dataset_name, train=True)
    testset  = get_dataset(dataset_name, train=False)

    n = len(trainset)
    idx = list(range(n)); random.shuffle(idx)
    half = n // 2
    victim_idx, shadow_pool_idx = idx[:half], idx[half:]

    def split_7_1_2(index_list):
        m = len(index_list)
        a = int(m*0.7); b = int(m*0.1)
        train_idx = index_list[:a]
        dev_idx   = index_list[a:a+b]
        test_idx  = index_list[a+b:]
        return Subset(trainset, train_idx), Subset(trainset, dev_idx), Subset(trainset, test_idx)

    victim_train_dataset, victim_dev_dataset, victim_test_dataset = split_7_1_2(victim_idx)

    per = len(shadow_pool_idx)//shadow_num
    attack_split_list = []
    shadow_train_list = []
    for s in range(shadow_num):
        part_idx = shadow_pool_idx[s*per:(s+1)*per]
        tr, dv, te = split_7_1_2(part_idx)
        attack_split_list.append((tr, dv, te))
        shadow_train_list.append(part_idx)

    save_folder = f"result/{dataset_name}_{model_name}"
    os.makedirs(save_folder, exist_ok=True)
    path = f"{save_folder}/data_prepare.pkl"
    with open(path, "wb") as f:
        pickle.dump((victim_idx, victim_train_dataset, victim_dev_dataset, victim_test_dataset,
                     attack_split_list, shadow_train_list), f)
    print("saved:", path)


if __name__ == "__main__":
    main()

