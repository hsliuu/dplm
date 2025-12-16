# 复制 SimpleFold 的 FlowMatchingLoss 逻辑
def compute_flow_matching_loss(v_pred, v_target):
    """
    Args:
        v_pred: FlowNetwork 预测的速度场
        v_target: 目标速度场 (由 t, X_0, X_t 计算)
    Returns:
        Loss: ||v_pred - v_target||^2
    """
    return torch.mean(torch.square(v_pred - v_target))

# SimpleFold的ODE Integrator (用于推理)
class ODEIntegrator:
    def __init__(self, step_size, flow_network):
        self.dt = step_size
        self.flow_network = flow_network

    def step(self, X_t, t, **conditions):
        """
        使用 Euler 或 Runge-Kutta 方法积分一步
        """
        # 假设 t 也是条件的一部分
        v_t = self.flow_network(X_t, **conditions)
        X_t_next = X_t - v_t * self.dt # 流匹配通常是t=1到t=0，这里是反向积分
        return X_t_next