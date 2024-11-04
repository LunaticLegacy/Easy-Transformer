import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from jax.example_libraries.optimizers import optimizer

from transformer_block import *

if __name__ == "__main__":
    device = torch.device("cuda:0")
    # 下面这个东西具有4个词向量，且每个词向量有6个维度。
    x: torch.tensor = torch.tensor(
        data=
        [
            [
                [1, 2, 3, 1, 2, 1],
                [1, 3, 2, 2, 1, 2],
                [3, 2, 1, 2, 1, 2],
                [1, 1, 4, 5, 1, 4]
            ],
            [
                [3, 3, 1, 1, 3, 1],
                [1, 0, 1, 2, 2, 2],
                [2, 3, 4, 2, 1, 1],
                [2, 2, 4, 1, 3, 4]
            ]
        ],
        dtype=torch.float32
    ).to(device)

    y: torch.Tensor = torch.tensor(
        data=
        [
            [
                [1, 2, 3, 1, 3, 1],
                [1, 3, 2, 2, 2, 1],
                [3, 2, 1, 2, 0, 1],
                [1, 1, 4, 5, 1, 3]
            ],
            [
                [1, 2, 3, 1, 3, 1],
                [1, 3, 2, 2, 2, 1],
                [3, 2, 1, 2, 0, 1],
                [1, 1, 4, 5, 1, 3]
            ]
        ],
        dtype=torch.float32
    ).to(device)

    # 定义模型
    trans = Transformer(
        encoder_block_num=4,
        decoder_block_num=4,
        attention_num_heads=4,
        max_length=4,
        dim=6,
        device=device
    )

    # 训练模型
    trans.train_workflow(encoder_in=x, decoder_in=y, iteration_times=10000)

    # 测试模型
    # 测试用输入
    z: torch.Tensor = torch.tensor(
        data=
        [
            [
                [3, 3, 1, 1, 3, 1],
                [1, 0, 1, 2, 2, 2],
                [2, 3, 4, 2, 1, 1],
                [2, 2, 4, 1, 3, 4]
            ]
        ],
        dtype=torch.float32
    ).to(device)

    result = trans.inference_workflow(z, torch.tensor([1, 2, 3, 1, 3, 1]).to(device))
    print(f"Result: \n{result}")
    print(f"Result shape: {result.shape}")
    # print(f"Loss: \n{trans.loss_list}")

    plt.plot(trans.loss_list)
    plt.show()
