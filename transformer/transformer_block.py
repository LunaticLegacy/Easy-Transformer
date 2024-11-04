import torch
import torch.nn as nn

from self_attention import *
from tqdm import tqdm


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
            ).to(device) for _ in range(block_num - 2)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 处理可变长度序列，并生成掩膜。
        x, _ = length_adjust(x, self.max_length)

        # 多头自注意力机制。
        x = torch.add(x, self.attention(x))  # 这里也需要一个残差链接

        # 线性层。
        for linear_block in self.linear_blocks:
            x = torch.add(x, linear_block(x))  # 残差连接

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
            ).to(device) for _ in range(block_num - 3)
        ]

    def forward(self,
                x: torch.Tensor,
                encoder_result: torch.Tensor = None,
                self_circulation: bool = False) -> torch.Tensor:
        """

        :param x: 输入的参数。
        :param encoder_result: 输入数据的结果。

        以下两个变量仅当在解码器的自循环迭代推理阶段才会使用。
        :param self_circulation: 解码器阶段将使用的内容，将其设置为True以告知这是自循环阶段。
        :return:
        """
        # 截断！
        x, _ = length_adjust(x, self.max_length)

        # 初始化时间编码掩膜
        time_mask: torch.Tensor = None

        # 在解码器推理过程的自循环时使用的，每帧使用一次，以保证输出。
        if self_circulation:
            time_mask = generate_causal_mask(x.shape[1]).to(self.device)

        # 第一层多头自注意力机制。
        x = torch.add(x, self.attention1(x, time_mask))

        # 如果有来自编码器的信息，则将编码器的信息直接叠加到x上。
        if encoder_result is not None:
            x = torch.add(x, encoder_result)

        # 第二层多头自注意力机制。
        x = torch.add(x, self.attention2(x, time_mask))

        # 接下来的每一层。
        for linear_block in self.linear_blocks:
            x = torch.add(x, linear_block(x))  # 残差连接

        return x


class Transformer(nn.Module):
    """
    一个被封装起的、可以实际使用的Transformer模型。
    """

    def __init__(self,
                 encoder_block_num: int,
                 decoder_block_num: int,
                 attention_num_heads: int,
                 max_length: int,
                 dim: int,
                 device: torch.device = torch.device("cpu")
                 ):
        """
        需传递变量：
        :param encoder_block_num: 编码器块数。
        :param decoder_block_num: 解码器块数。

        :param attention_num_heads: 多头自注意力机制使用的头数。
        :param max_length: 最大上下文长度。
        :param dim: 维度数。
            输入的数据是一张有n个词向量的（每个词向量的维度数为m）词组。
            所以，这个东西规定了最大的n。如果n小于这个数字，则这个机制会自动将n补全。
            输入的维度数则被写入下方的dim。

        :param device: 使用的设备
        """
        super().__init__()
        # 初始化所有变量
        self.encoder_block_num = encoder_block_num
        self.decoder_block_num = decoder_block_num

        self.attention_num_heads = attention_num_heads
        self.max_length = max_length
        self.dim = dim
        self.device = device

        # 记录损失情况的东西
        self.loss_list: list[float] = []

        # 初始化编码器块和解码器块
        self.encoder = Encoder(self.encoder_block_num, self.attention_num_heads, self.max_length, self.dim, self.device)
        self.decoder = Decoder(self.decoder_block_num, self.attention_num_heads, self.max_length, self.dim, self.device)
        # 并序列化
        self.transformer_workflow: nn.Sequential = nn.Sequential(
            self.encoder,
            self.decoder
        )

        # 初始化本模型内的词汇表大小。
        self.vocab_size: int = 0

        # 优化器，这里采用Adam优化器。
        self.optimizer = torch.optim.Adam(self.transformer_workflow.parameters(), lr=4e-4)

    def train_forward(self,
                      encoder_in: torch.Tensor,
                      decoder_in: torch.Tensor,
                      ) -> torch.Tensor:
        """
        模型训练过程的前向传播。
        :param encoder_in: 输入到编码器里的内容。
        :param decoder_in: 输入到解码器里的内容。
        :return: 结果。
        """
        # 切换到训练模式。
        self.encoder.train()
        self.decoder.train()

        encoder_result = self.encoder.forward(x=encoder_in)
        decoder_result = self.decoder.forward(x=decoder_in, encoder_result=encoder_result)
        return decoder_result

    def train_backward(self,
                       decoder_in: torch.Tensor,
                       decoder_result: torch.Tensor
                       ) -> torch.Tensor:
        """
        模型训练过程的反向传播。
        如果可以，请在实际应用时覆写该方法。
        :param decoder_in: 解码器的输入，作为真实值。(batch_size, self.max_length, self.dim)
        :param decoder_result: 解码器的输出。(batch_size, self.max_length, self.dim)
        :return: 本次训练的损失情况。
        """
        # 定义损失函数，交叉熵损失函数。
        loss_fn = torch.nn.CrossEntropyLoss()

        # 调整输出形状
        # decoder_result = decoder_result.view(-1)  # 变形为 (batch_size * max_length, vocab_size)
        # decoder_in = decoder_in.view(-1)  # 变形为 (batch_size * max_length)

        # 计算本批次的损失（注意顺序：输出在前，标签在后）
        loss_value: torch.Tensor = loss_fn(decoder_result, decoder_in)

        # 重置梯度
        self.optimizer.zero_grad()
        # 反向传播
        loss_value.backward()
        # 更新参数
        self.optimizer.step()

        return loss_value

    def train_workflow(self,
                       encoder_in: torch.Tensor,
                       decoder_in: torch.Tensor,
                       iteration_times: int
                       ) -> None:
        """
        训练模型时，对单组训练数据的反复迭代。
        :param encoder_in: 输入到编码器里的内容。
        :param decoder_in: 输入到解码器里的内容，同时作为模型反向传播时的真实值。
        :param iteration_times: 循环迭代多少次。
        :return: 不返回任何东西。
        """
        # 获取唯一单词索引
        unique_words = set(encoder_in.flatten().tolist()).union(set(decoder_in.flatten().tolist()))
        # 计算词汇表大小
        self.vocab_size = len(unique_words)

        for _ in tqdm(range(iteration_times), desc='Training...'):
            forward: torch.Tensor = self.train_forward(encoder_in, decoder_in)
            loss: float = self.train_backward(decoder_in, forward).item()
            self.loss_list.append(round(loss, 6))

    def inference_workflow(self,
                           x: torch.Tensor,
                           initial_marker: torch.Tensor,
                           ) -> torch.Tensor:
        """
        在推理阶段，该模型的全部循环过程。
        :param x: 为编码器输入的内容。
        :param initial_marker: 在训练期间使用的，作为“序列开始”的序列。
        :return:
        """
        # 切换到评估模式。
        self.encoder.eval()
        self.decoder.eval()

        # 对这个x进行长度补全。
        x, _ = length_adjust(x, self.max_length)

        # 初始序列，这些东西为全0内容。
        initial_sequence: torch.Tensor = (torch.zeros((1, self.max_length, self.dim), dtype=torch.float32)
                                          .to(self.device))

        # 填充初始序列的第一个词向量位。
        initial_sequence[:, 0] = torch.tensor(initial_marker, dtype=torch.float32).clone().detach()

        # 将初始序列进行编码，手动执行第一次循环。
        encoder_result: torch.Tensor = self.encoder.forward(x=x)
        decoder_result: torch.Tensor = self.decoder.forward(x=initial_sequence,
                                                            encoder_result=encoder_result,
                                                            self_circulation=True)

        # 获取预测结果
        prediction = decoder_result[:, -1]  # 获取当前时间步的预测
        # 将预测结果添加到初始序列中
        initial_sequence[:, 1] = prediction

        # 然后进入自循环
        for i in tqdm(range(1, self.max_length), desc="Generating..."):
            decoder_result = self.decoder.forward(x=decoder_result)
            # 获取预测结果
            prediction = decoder_result[:, -1]  # 获取当前时间步的预测
            # 将预测结果添加到初始序列中
            initial_sequence[:, i] = prediction

        return initial_sequence


def generate_causal_mask(seq_length: int) -> torch.Tensor:
    """
    创建一个因果掩膜，形状为 (seq_length, seq_length)。
    :param seq_length: 序列长度。
    :return: 掩膜张量，包含有效位置为0，未来位置为-inf。
    """
    mask = torch.zeros(seq_length, seq_length)  # 初始化为0
    mask = mask.masked_fill(torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool(), float('-inf'))
    return mask


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

    encoder = Encoder(
        block_num=8,
        attention_num_heads=4,
        max_length=3,
        dim=6,
        device=device
    )

    decoder = Decoder(
        block_num=8,
        attention_num_heads=4,
        max_length=3,
        dim=6,
        device=device
    )

    encoder_result = encoder.forward(x=x)
    decoder_result = decoder.forward(x=x, encoder_result=encoder_result)

    # 反向传播
    loss_fn = torch.nn.MSELoss()
    # 计算损失
    loss_value = loss_fn(encoder_result, decoder_result)
    # 优化器
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

    # 重置梯度
    optimizer.zero_grad()
    # 反向传播
    loss_value.backward()
    # 更新参数
    optimizer.step()
