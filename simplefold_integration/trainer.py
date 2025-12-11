def training_step(model, batch):
    # 从 DataLoader 获取所有数据
    (seq_ids, target_seq, struct_X, target_bit, target_fine_cond, X_t, v_target, t) = batch
    
    # 1. LM 和 Bit Head 前向传播
    seq_logits, bit_logits, lm_hidden = model(
        input_seq_ids=seq_ids, 
        input_struct_X=X_t, # 输入是带噪声的结构
        target_t=t 
    )
    
    # --- A. 序列损失 ---
    seq_loss = F.cross_entropy(seq_logits.view(-1, V), target_seq.view(-1))
    
    # --- B. Bit 损失 ---
    bit_loss = F.cross_entropy(bit_logits.view(-1, 2), target_bit.view(-1))
    
    # --- C. 流匹配损失 ---
    # 2. Flow Network 前向传播
    v_pred = model.flow_network(
        X_t=X_t, 
        lm_hidden=lm_hidden.detach(), # 关键：不让FM损失的梯度流回LM，除非使用交叉损失
        fine_cond=target_fine_cond,
        sequence_mask=None
    )
    
    flow_matching_loss = compute_flow_matching_loss(v_pred, v_target)
    
    # --- D. 联合优化 ---
    lambda_bit = 1.0 # 超参数
    lambda_fm = 100.0 # 超参数
    
    total_loss = seq_loss + (lambda_bit * bit_loss) + (lambda_fm * flow_matching_loss)
    
    # 反向传播和优化器更新
    # ...
    return total_loss