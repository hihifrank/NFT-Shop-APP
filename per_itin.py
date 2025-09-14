#test#
"""
import pandas as pd
data = pd.read_csv('user_data_test.csv')
print(data.head())
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 載入數據
# 假設數據已存為 'user_data_test.csv'，這裡直接使用你的數據
data = pd.read_csv('user_data_test.csv')

# 提取特徵（排除 user_id）
features = data.iloc[:, 1:].to_numpy()  # 使用 .to_numpy() 代替 .values
print("Features shape:", features.shape)  # 應為 (2, 9)
print("Features:\n", features)  # 檢查數據內容

# 將第一筆數據轉為 float32，並檢查是否有 NaN 或無窮值
user_data_np = features[0:1].astype(np.float32)
if np.any(np.isnan(user_data_np)) or np.any(np.isinf(user_data_np)):
    raise ValueError("數據包含 NaN 或無窮值")
print("User data (numpy):\n", user_data_np)  # 檢查轉換後的數據

# 轉為 PyTorch 張量，明確指定 dtype
user_data = torch.tensor(user_data_np, dtype=torch.float32)
print("User data (tensor):\n", user_data)  # 檢查張量

# 定義推薦模型
class ItineraryRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 20)  # 輸入9維（根據數據特徵數），隱藏層20
        self.fc2 = nn.Linear(20, 3)  # 輸出3條行程選項的分數

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 創建模型
model = ItineraryRecommender()

# 訓練設置
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 假設目標（模擬最佳分數，實際用真標籤替換）
target = torch.tensor([[0.8, 0.5, 0.2]], dtype=torch.float32)  # 行程A最高分

# 訓練循環
for epoch in range(100):
    optimizer.zero_grad()
    output = model(user_data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# 最終輸出
final_output = model(user_data)
print("最終行程分數:", final_output)
