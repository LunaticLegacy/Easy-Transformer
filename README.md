目前进度：<br>
    ·Transformer模型本体：已完成。<br>
    ·Swin Transformer：正在工作。<br>
    ……（还有其他可能的Transformer）<br>
<br>
本集合里所有的代码均为“有手就行”型，不需要看什么纷繁复杂的README，只要套上Torch和Torch.nn就能用。<br>


文件档案：<br>
    ·Transformer：<br>
        一个可用的Transformer模型本体，使用方法已放在文档内部注释。<br>
        使用方法：<br>
        “训练阶段”，即模型训练阶段。在Transformer模型中，训练阶段需要同时使用编码器和解码器。<br>
        在编码器中输入你用于训练的输入内容，然后在解码器中输入。<br>
        示例代码如下：（这些代码都在usage.py内）<br>
```
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

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

     
```
