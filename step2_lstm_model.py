import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. 数据模拟与准备 (你需要替换为真实的合并数据) ---
# 假设我们已经把 nlp_features_ready.csv 和 Capital IQ 的 MOVE Index 合并了
# 创造一些假数据来演示流程
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=200)
divergence_scores = np.random.uniform(0.1, 0.5, 200) # 模拟 NLP 分数
move_index = np.random.normal(100, 15, 200) + (divergence_scores * 20) # 假设 NLP 分数影响波动率

df_model = pd.DataFrame({
    'Date': dates,
    'Divergence': divergence_scores,
    'MOVE': move_index
})

# --- 2. 数据预处理 (LSTM 对数值范围很敏感，必须归一化) ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_model[['Divergence', 'MOVE']])

def create_sequences(data, seq_length=5):
    """创建时间序列窗口: 用过去 5 天的数据预测下一天"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length] # 输入: 过去5天的特征
        y = data[i+seq_length, 1] # 输出: 第6天的 MOVE Index (列索引1)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 5
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# 转为 PyTorch Tensor
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y).view(-1, 1)

# --- 3. 定义 LSTM 模型 (High-Level Code) ---
class FedLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super(FedLSTM, self).__init__()
        self.hidden_size = hidden_size
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.2)
        # 全连接层输出预测值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, features)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

# --- 4. 训练模型 ---
model = FedLSTM(input_size=2) # 输入特征有2个: Divergence 和 MOVE
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("开始训练 LSTM 模型...")
epochs = 100
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.6f}')

# --- 5. 结果可视化 (画给老师看) ---
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('LSTM Training Loss (Convergence)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

print("模型训练完成！可以将此代码截图放入报告的 Methodology 章节。")