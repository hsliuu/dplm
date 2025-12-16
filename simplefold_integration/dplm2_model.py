# dplm/model/dplm2_model.py

from dplm.simplefold_integration.flow_network import SimpleFoldFlowNetwork

class DPLM2HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        #保留Transformer Backbone
        self.transformer = DPLM2Transformer(config)
        self.seq_head = nn.Linear(config.hidden_dim, config.vocab_size)
        
        #这个暂时有点问题
        # 添加Bit Head (预测32个结构 bit)
        # 假设L是序列长度，32是bit数，BIT_VOCAB_SIZE=2
        #self.bit_head = nn.Linear(config.hidden_dim, 32 * config.bit_vocab_size) 
        
        # 添加Flow Network
        self.flow_network = SimpleFoldFlowNetwork(
            lm_hidden_dim=config.hidden_dim,
            fine_cond_dim=config.fine_cond_dim, # 需要新的配置项
            struct_dim=3 
        )
        
        # 结构编码器—结构优化序列
        self.structure_encoder = StructureGNN(input_dim=3, output_dim=config.hidden_dim)
        
    def forward(self, input_seq_ids, input_struct_X, target_t=None):
        """
        输入: 序列 token, 当前结构坐标 X, 训练时目标时间 t
        """
        
        # 结构编码：将结构坐标X编码成特征E_X
        # 假设结构编码器输出 (B, L, H)
        E_X = self.structure_encoder(input_struct_X) 
        
        # Transformer前向传播 (在输入中加入结构特征)
        # 将E_X融入到序列嵌入中，例如通过相加：
        seq_embeddings = self.transformer.embeddings(input_seq_ids) + E_X
        # Transformer执行：返回隐藏状态
        lm_hidden = self.transformer(seq_embeddings) 
        
        # 序列头输出
        seq_logits = self.seq_head(lm_hidden)
        
        # Bit头输出
        # 停用
        # bit_logits = self.bit_head(lm_hidden).view(-1, 32, config.bit_vocab_size)
        
        # 流匹配在训练时计算损失
        v_pred = None
        if target_t is not None:
            # 假设target_fine_cond 已在数据加载器中预计算
            # 这里的target_struct_X, target_fine_cond, v_target从DataLoader 传入
            
            # 训练逻辑
            
            v_pred = self.flow_network(
                 X_t=target_struct_X_t, # 扩散噪声后的结构
                 lm_hidden=lm_hidden.detach(), # 阻止梯度流回LM
                 fine_cond=target_fine_cond,
                 sequence_mask=None
             )
            flow_matching_loss = compute_flow_matching_loss(v_pred, v_target)
            
            pass
            
        return seq_logits, bit_logits, lm_hidden