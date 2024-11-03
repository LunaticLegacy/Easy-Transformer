import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    """
    单头自注意力机制。
    """

    def __init__(self,
                 max_length: int,
                 dim: int,
                 device: torch.device = torch.device("cpu")
                 ):
        """
        一个自注意力头使用的变量。
        :param max_length: 最大上下文长度。
        :param dim: 维度数。
            输入的数据是一张有n个词向量的（每个词向量的维度数为m）词组。这个东西规定了最大的n。
                ·如果n小于这个数字，则这个机制会自动将n补全，并生成指定长度的掩膜。
                ·如果n大于这个数字，则会截断到n。
            该机制的作用：用于让模型可处理可变长度序列。
            输入的维度数m为变量dim。
        :param device: 使用的设备。
        """
        super().__init__()
        self.max_length = max_length
        self.dim = dim
        self.device = device

        # 初始化Q、K和V的矩阵。因为这三个东西实际上都是线性代数操作，所以使用linear。
        # 其中，Q = x * Weight.Q，K和V同理。
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入的数据，形状为(batch_size, max_length, dim)。
        :return: 输出的数据，形状为(batch_size, max_length, dim)。
        """
        # 处理可变长度序列，返回掩膜。
        x, mask = length_adjust(x, self.max_length)

        # softmax ( (Q * K.T) / sqrt(dim) ) * V = Z
        Q_x = self.Q(x)
        K_x = self.K(x)
        V_x = self.V(x)

        # 计算 QK^T
        QK_T = torch.matmul(Q_x, K_x.transpose(-2, -1))
        # 计算 QK^T / sqrt(dim)
        scaled_QK_T = QK_T / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        # 应用 softmax
        attention_weights = self.softmax(scaled_QK_T)
        # 计算最终输出
        output = torch.matmul(attention_weights, V_x)

        # 如果存在掩膜：
        if mask is not None:
            output = output * mask

        return output


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制。
    实际上这个东西就是将单头自注意力机制进行融合。
    """

    def __init__(self,
                 num_heads: int,
                 max_length: int,
                 dim: int,
                 device: torch.device = torch.device("cpu")
                 ):
        """
        新增变量：
        :param num_heads: 注意力头数。

        需传递变量：
        :param max_length: 最大上下文长度。
        :param dim: 维度数。
            输入的数据是一张有n个词向量的（每个词向量的维度数为m）词组。这个东西规定了最大的n。
                ·如果n小于这个数字，则这个机制会自动将n补全，并生成指定长度的掩膜。
                ·如果n大于这个数字，则会截断到n。
            该机制的作用：用于让模型可处理可变长度序列。
            输入的维度数m为变量dim。
        :param device: 使用的设备。
        """
        super().__init__()

        # 初始化所有变量
        self.num_heads = num_heads
        self.max_length = max_length
        self.dim = dim
        self.device = device

        # 定义自注意力头
        self.headlist: list[AttentionHead] = [AttentionHead(self.max_length, self.dim, self.device)
                                              for _ in range(num_heads)]

        # 算法：将8个头输出的内容，concat，然后乘以一个大矩阵（用Linear），使其还原为原本模样。
        self.concat_matrix = nn.Linear(num_heads * dim, dim)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        # 处理可变长度序列，返回掩膜。
        output, mask = length_adjust(x, self.max_length)

        head_outputs: list[torch.Tensor] = [head(x) for head in self.headlist]
        concat_output = torch.cat(head_outputs, dim=-1)
        output = self.concat_matrix(concat_output)

        # 掩膜
        if mask is not None:
            output = torch.mul(output, mask)

        return output


def length_adjust(x: torch.Tensor,
                  max_length: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    将输入的序列进行长度调整。
    :param x: 输入的序列，形状为(batch_size, max_length, dim)。
    :param max_length: 序列的最大长度。
    :return: 一个列表。
        其中第一个内容是调整后的序列，形状为(batch_size, max_length, dim)。
        第二个内容为掩膜，如果不生成掩膜则返回None。
    """
    batch_size, seq_length, dim = x.shape

    mask: torch.Tensor | None = None  # 初始化掩膜变量
    if x.shape[1] < max_length:
        # 生成掩膜
        mask = torch.zeros(x.shape[0],
                           max_length,
                           x.shape[2],
                           dtype=torch.bool,
                           device=x.device)
        mask[:, :x.shape[1]] = True

        # 创建一个新的张量，用于填充
        padded_x = torch.zeros(batch_size,
                               max_length,
                               dim,
                               device=x.device)

        # 将原始数据复制到填充张量中
        padded_x[:, :seq_length, :] = x

        # 实施掩膜
        x = torch.mul(x, mask)

    elif x.shape[1] > max_length:
        x = x[:, :max_length]

    return (x, mask)


if __name__ == "__main__":  # 测试例。没错我根本懒得写单元测试而是直接这样测试。
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

    head = MultiHeadAttention(
        max_length=6,
        dim=6,
        num_heads=4,
        device=device
    )

    head_forward = head.forward(x)

    print(head_forward)
    print(head_forward.shape)
