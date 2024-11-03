import torch
import torch.nn as nn
from self_attention import *

class Encoder(nn.Module):
    """
    定义一个基础的编码器。
    根据Transformer模型曰过的东西，只有第一层需要用到多头自注意力机制，其他头只需要线性+残差即可。
    """
    def __init__(self,
                 block_num: int,
                 attention_num_heads: int,
                 max_length: int,
                 dim: int,
                 device: torch.device = torch.device("cpu")
                 ):
        """
        新增变量：
        :param block_num: 这个编码器有多少块。

        需传递变量：
        :param attention_num_heads: 多头自注意力机制使用的头数，传递到多头自注意力机制时为num_heads。
        :param max_length: 最大上下文长度。
        :param dim: 维度数。
            输入的数据是一张有n个词向量的（每个词向量的维度数为m）词组。
            所以，这个东西规定了最大的n。如果n小于这个数字，则这个机制会自动将n补全。
            输入的维度数则被写入下方的dim。
            （这个东西我在self_attention.py里解释过，这里不需要再继续解释了）
        :param device: 使用的设备
        """
        super().__init__()
        # 检查变量
        assert block_num >= 2, \
            "block_num in encoder must be more than 2, just learn the principle of transformer before you come back."
        # -- 初始化所有变量 --
        self.block_num = block_num
        self.num_heads = attention_num_heads
        self.max_length = max_length
        self.dim = dim
        self.device = device

        # -- 初始化块 --
        # 自注意力块。
        self.attention = MultiHeadAttention(self.num_heads, self.max_length, self.dim, self.device)  # 多头自注意力机制。

        # 然后，对于第二层后的每一个block，各写一个Linear层。残差将在forward方法中手动实现。
        self.linear_blocks: list[nn.Sequential] = [
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            ) for _ in range(block_num - 2)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 截断！
        x, mask = length_adjust(x, self.max_length)

        # 多头自注意力机制。
        x += self.attention(x)  # 这里也需要一个残差链接
        if mask is not None:  # 掩膜
            x = x * mask

        # 线性层。
        for linear_block in self.linear_blocks:
            x = x + linear_block(x)  # 残差连接
            if mask is not None:  # 掩膜
                x = x * mask

        return x


class Decoder(nn.Module):
    """
    定义一个基础的解码器。
    根据Transformer模型曰过的东西，这个东西的前两层均需要使用多头自注意力机制。
    其中，第一个多头自注意力机制用于将解码器的输入数据进行编码，第二个多头自注意力机制用途如下：
        ·如果编码器给出数据（训练期间）：第一个多头自注意力编码器（块）给出的数据 + 编码器的最终输出，将这个东西输入后进行处理。
        ·如果编码器没给数据（推理，即让模型输出内容）：直接锤第一个多头自注意力编码器（块）给出的数据。
    其他头只需要线性+残差即可。
    """

    def __init__(self,
                 block_num: int,
                 attention_num_heads: int,
                 max_length: int,
                 dim: int,
                 device: torch.device = torch.device("cpu")
                 ):
        """
        新增变量：
        :param block_num: 这个解码器有多少块。

        需传递变量：
        :param attention_num_heads: 多头自注意力机制使用的头数，传递到多头自注意力机制时为num_heads。
        :param max_length: 最大上下文长度。
        :param dim: 维度数。
            输入的数据是一张有n个词向量的（每个词向量的维度数为m）词组。
            所以，这个东西规定了最大的n。如果n小于这个数字，则这个机制会自动将n补全。
            输入的维度数则被写入下方的dim。
            （这个东西我在self_attention.py里解释过，这里不需要再继续解释了）
        :param device: 使用的设备
        """
        super().__init__()
        assert block_num >= 3, \
            "block_num in decoder must be more than 3, just learn the principle of transformer before you come back."
        # -- 初始化所有变量 --
        self.block_num = block_num
        self.num_heads = attention_num_heads
        self.max_length = max_length
        self.dim = dim
        self.device = device

        # -- 初始化块 --
        # 2块自注意力块
        self.attention1 = MultiHeadAttention(self.num_heads, self.max_length, self.dim, self.device)  # 第一层多头自注意力机制。
        self.attention2 = MultiHeadAttention(self.num_heads, self.max_length, self.dim, self.device)  # 第二层多头自注意力机制。

        # 然后，对于第二层后的每一个block，各写一个Linear层。
        self.linear_blocks: list[nn.Sequential] = [
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            ) for _ in range(block_num - 3)
        ]

    def forward(self,
                x: torch.Tensor,
                encoder_input: torch.Tensor = None) -> torch.Tensor:
        # 截断！
        x, mask = length_adjust(x, self.max_length)

        # 第一层多头自注意力机制。
        x += self.attention1(x)
        if mask is not None:  # 掩膜
            x = x * mask

        # 如果有来自编码器的信息，则将编码器的信息直接叠加到x上。
        if encoder_input is not None:
            x = x + encoder_input

        if mask is not None:  # 掩膜
            x = x * mask

        # 第二层多头自注意力机制。
        x += self.attention2(x)
        if mask is not None:  # 掩膜
            x = x * mask

        # 接下来的每一层。
        for linear_block in self.linear_blocks:
            x = x + linear_block(x)  # 残差连接
            if mask is not None:  # 掩膜
                x = x * mask

        return x

if __name__ == "__main__":
    device = torch.device("cpu")

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
          [1, 2, 3, 1, 2, 1],
          [1, 3, 2, 2, 1, 2],
          [3, 2, 1, 2, 1, 2],
          [1, 1, 4, 5, 1, 4]],
         ],
        dtype=torch.float32
    )

    head = Encoder(
        block_num=8,
        attention_num_heads=4,
        max_length=3,
        dim=6,
        device=device
    )

    head_forward = head.forward(x)

    print(head_forward)
    print(head_forward.shape)

