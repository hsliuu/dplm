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

"""  # dplm/training/trainer.py (简化伪代码)

def training_step(model, batch, epoch, total_epochs):
    # 假设 batch 包含所有所需数据: real_seq, real_coords, t, target_velocity, target_bit 等
    
    real_seq, real_coords, t, target_velocity, target_bit = batch # 从 loader 中解包
    
    # 真实坐标加噪，得到 coords_t
    # 注意：需要确保 noise() 函数能正确计算 coords_t，
    # 并且 noise 函数和 SimpleFold 的噪声机制兼容
    coords_t = noise(real_coords, t) 
    
    # 1. LM 同时预测序列 + 32 个 bit
    # LM 接收 seq_tokens（可能是带 MASK 的输入或自回归输入）
    seq_logits, bit_logits, lm_hidden = model.lm(real_seq)
    
    # 2. bit-wise → RESDIFF 补高清残差（DPLM-2.1 独有逻辑）
    # bits_to_zquant 和 resdiff 应该是 model.module 或一个单独的预处理模块
    z_quant   = model.bits_to_zquant(bit_logits)
    residual  = model.resdiff(lm_hidden, z_quant, t)
    fine_cond = z_quant + residual                 
    
    # 3. FlowNetwork 接收双条件
    # 注意：FlowNetwork 应该作为 model 的一个子模块
    velocity = model.flow_network(lm_hidden, fine_cond, coords_t, t)
    
    # 4. 流匹配损失
    # 注意：流匹配损失的目标是速度场 v_target，而不是简单的 real_coords - coords_t。
    # SimpleFold 的损失通常是 ||v_pred - v_target||^2，v_target 需要根据 t 和 real_coords 计算。
    # 假设 target_velocity 已预计算或在 SimpleFold 的损失函数内部计算。
    fm_loss = flow_matching_loss(velocity, target_velocity)
    
    # 5. 序列损失和 Bit 损失 (此处缺失，但必须添加)
    seq_loss = compute_sequence_loss(seq_logits, target_seq)
    bit_loss = compute_bit_loss(bit_logits, target_bit)
    
    # 6. 总损失
    total_loss = fm_loss + seq_loss * lambda_seq + bit_loss * lambda_bit
    
    # 7. 加 folding SFT (结构微调)
    if epoch / total_epochs > 0.9:
        # 在 Flow Network 上计算一个简单的 MSE (结构微调 SFT)
        # 假设 Flow Network 能够预测一个去噪后的结构 X_0，或者这里需要一个单独的结构预测头
        # **注意：您这里用 flow_network(...) 计算 MSE 的方式，需要 Flow Network 有一个结构输出头。**
        
        # 简化处理：假设 Flow Network 可以通过积分得到 X_0 预测
        # 或者使用一个额外的结构预测头 model.struct_head()
        predicted_coords_0 = model.struct_head(velocity) # 伪代码：需要根据实际模型调整
        sft_loss = 0.1 * mse(predicted_coords_0, real_coords)
        total_loss += sft_loss
        
    # 反向传播等...
    # total_loss.backward()
    # optimizer.step()
    
    return total_loss """