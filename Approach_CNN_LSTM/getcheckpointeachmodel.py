import os

list_dir = ["Flex_Model_noFC_False_False", "Flex_Model_noFC_False_True", "Flex_Model_noFC_True_False"]

root = "../Historyfromserver"


def make_cli(dir):
    num_cnn = dir[23: 24]
    num_lstm = dir[25: 26]
    is_resnet = dir[27: 32]
    is_bidirectional = dir[33: 38]
    print(num_cnn, num_lstm, is_resnet, is_bidirectional)
    if is_resnet == "True_":
        if is_bidirectional == "True":
            rs = f"python on_Testset.py --model_type Flex_Model_noFC --filedata ./data/eegsplit_bytime --splits_path ./data/block_splits_by_image_all.pth --model_path ./history/{dir}/checkpoint/checkpoint_best.pth --save_file ./history/runtime_inference.txt --num_cnn_layers {num_cnn} --num_lstm_layers {num_lstm} --is_resnet 1 --is_bidirectional 1"
        else:
            rs = f"python on_Testset.py --model_type Flex_Model_noFC --filedata ./data/eegsplit_bytime --splits_path ./data/block_splits_by_image_all.pth --model_path ./history/{dir}/checkpoint/checkpoint_best.pth --save_file ./history/runtime_inference.txt --num_cnn_layers {num_cnn} --num_lstm_layers {num_lstm} --is_resnet 1"
    else:
        if is_bidirectional == "True":
            rs = f"python on_Testset.py --model_type Flex_Model_noFC --filedata ./data/eegsplit_bytime --splits_path ./data/block_splits_by_image_all.pth --model_path ./history/{dir}/checkpoint/checkpoint_best.pth --save_file ./history/runtime_inference.txt --num_cnn_layers {num_cnn} --num_lstm_layers {num_lstm} --is_bidirectional 1"
        else:
            rs = f"python on_Testset.py --model_type Flex_Model_noFC --filedata ./data/eegsplit_bytime --splits_path ./data/block_splits_by_image_all.pth --model_path ./history/{dir}/checkpoint/checkpoint_best.pth --save_file ./history/runtime_inference.txt --num_cnn_layers {num_cnn} --num_lstm_layers {num_lstm}"

    return rs


rs = []

for dir in list_dir:
    path = os.path.join(root, dir)
    list_sub_dir = os.listdir(path)
    print(dir)
    for sub_dir in list_sub_dir:
        print(make_cli(sub_dir))
        rs.append(make_cli(sub_dir))
cnt = 1
with open("runall.txt", "a") as f:
    for r in rs:
        f.write(f"{r} ; ")
        if cnt % 5 == 0:
            f.write("\n")
        cnt += 1
