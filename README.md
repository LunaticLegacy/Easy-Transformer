目前进度：<br>
    ·Transformer模型本体：已完成。<br>
    ·Swin Transformer：未完成。<br>
    ……（还有其他可能的Transformer）<br>
<br>
本集合里所有的代码均为“有手就行”型，不需要看什么纷繁复杂的README，只要套上Torch和Torch.nn就能用。<br>

稍等一下，等我重新上传一下文件啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊

文件档案：<br>
    ·GraduTransformer：<br>
        一个可用的Transformer模型本体，需要使用时请自行输入编码器和解码器。<br>
        使用方法：<br>
        “训练阶段”，即模型训练阶段。在Transformer模型中，训练阶段需要同时使用编码器和解码器。<br>
        在编码器中输入你用于训练的输入内容，然后在解码器中输入。<br>
        示例代码如下：<br>
```
        
    from transformer_block import Encoder, Decoder
    from tqdm import tqdm  # 一个可视化循环进度的库

    if __name__ == "__main__":
        device = torch.device("cpu")

        # 下面这个东西中，每一句均具有4个词向量，且每个词向量有6个维度，共2句。
        # 形状为(batch_size, max_length, dim)，对应(句数, 词数, 词向量维度数)。
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
        )  # 示例数据，这个东西名叫x。

        head = Encoder(
            block_num=8,
            attention_num_heads=4,
            max_length=3,
            dim=6,
            device=device
        )  # 编码器

        decoder = Decoder(
            block_num=8,
            attention_num_heads=4,
            max_length=3,
            dim=6,
            device=device
        )  # 解码器

        # 反向传播，这里使用均方根误差损失函数。
        loss_fn = torch.nn.MSELoss()
        # 优化器，我喜欢使用Adam优化器
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)


        # 将以下内容扔给循环里，让模型进行训练。
        for _ in tqdm(range(100), desc="Training..."):
            # -- 前向传播 --
            encoder_result = encoder.forward(x=x)
            decoder_result = decoder.forward(x=x, encoder_result=encoder_result)  # 在推理阶段，解码器需要输入来自编码器的结果。

            # -- 反向传播 --        
            # 计算损失
            loss_value = loss_fn(encoder_result, decoder_result)        
            # 重置梯度
            optimizer.zero_grad()
            # 反向传播
            loss_value.backward()
            # 更新参数
            optimizer.step()

        
```
