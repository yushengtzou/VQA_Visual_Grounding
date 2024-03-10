'''
==========================================================================
 *
 *       Filename:  model.py
 *
 *    Description:  Use CLIP model API
 *
 *        Version:  1.0
 *        Created:  2024/03/11
 *       Revision:  none
 *       Compiler:  
 *
 *         Author:  鄒雨笙 
 *   Organization:  
 *
 * ==========================================================================
'''


# 引入模組
import torch
import torch.nn as nn


# ConvBlock 類別
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        # 定義卷積層，設定輸入輸出通道數、卷積核大小、步長、邊界填充等
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        # 定義批量標準化層，其通道數與卷積層的輸出通道數相同
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 將輸入x通過卷積層後再通過批量標準化層，並返回結果
        return self.bn(self.c(x))


# SE Block 類別
class SEBlock(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()
        # 全局平均池化，壓縮空間維度至1x1
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        # 第一個全連接層，降低維度以減少參數量
        self.fc1 = nn.Linear(C, C//r)
        # 第二個全連接層，恢復維度
        self.fc2 = nn.Linear(C//r, C)
        # ReLU激活函數
        self.relu = nn.ReLU()
        # Sigmoid激活函數，用於生成通道注意力權重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 的形狀：[N, C, H, W]
        f = self.globpool(x)  # 對特徵圖進行全局平均池化
        f = torch.flatten(f, 1)  # 展平特徵圖
        f = self.relu(self.fc1(f))  # 通過第一個全連接層和ReLU激活函數
        # 通過第二個全連接層和Sigmoid激活函數
        f = self.sigmoid(self.fc2(f))  
        # f 的形狀：[N, C]
        f = f[:,:,None,None]  # 調整形狀以匹配輸入x的形狀
        # f 的形狀：[N, C, 1, 1]
        return x * f  # 將計算得到的通道注意力權重乘以原始輸入


# Bottleneck ResNeXt ResidualBlock 結合 
# Squeeze-and-Excitation (SE) 模塊類別
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, first=False, cardinality=32):
    super().__init__()
    self.C = cardinality
    self.downsample = stride == 2 or first  # 判斷是否需要下採樣來匹配維度或者是第一個塊
    res_channels = out_channels // 2  # 計算殘差塊中間層的通道數

    # 定義殘差塊的卷積層，使用卡迪納利特技術（cardinality，即分組卷積）
    self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)  # 第一個卷積層，1x1卷積用於通道降維
    self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1, self.C)  # 第二個卷積層，3x3卷積，可能包含stride和分組
    self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)  # 第三個卷積層，1x1卷積用於通道升維
    self.relu = nn.ReLU()
    self.seblock = SEBlock(out_channels)  # Squeeze-and-Excitation模塊，用於通道注意力機制

    if self.downsample:
      # 如果需要下採樣，通過1x1卷積調整輸入x的維度
      self.p = ConvBlock(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):
        # 前向傳播過程
        f = self.relu(self.c1(x))  # 通過第一層卷積和ReLU激活
        f = self.relu(self.c2(f))  # 通過第二層卷積和ReLU激活
        f = self.c3(f)  # 通過第三層卷積
        f = self.seblock(f)  # 通過SE模塊進行通道加權

        if self.downsample:
          x = self.p(x)  # 如果需要，對輸入x進行下採樣

        h = self.relu(torch.add(f, x))  # 將轉換後的特徵f和原始輸入x相加，然後通過ReLU激活

        return h


# 定義一個簡單的網絡類別，繼承自 nn.Module
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 初始化一個殘差塊，設定輸入通道為3（對於RGB圖像），輸出通道為64，步長為1
        # 這裡 'first=True' 表示這是第一個殘差塊，可能需要進行一些特別的處理，如維度匹配
        self.layer1 = ResidualBlock(in_channels=3, out_channels=64, stride=1, first=True)

    def forward(self, x):
        # 定義網絡的前向傳播路徑
        # 將輸入 'x' 通過先前定義的殘差塊 'layer1'
        x = self.layer1(x)
        # 返回經過殘差塊處理後的輸出 'x'
        return x



if __name__ == '__main__':
    # 初始化 SimpleNet 模型
    model = SimpleNet()

    # 創建一個假設的輸入張量，模擬一個批次大小為4的RGB圖像批次，
    # 每個圖像大小為28x28
    # 張量形狀說明：[批次大小, 通道數, 高度, 寬度]
    input_tensor = torch.rand(4, 3, 28, 28)

    # 將輸入張量 'input_tensor' 通過模型 'model' 進行前向傳播
    output_tensor = model(input_tensor)

    # 輸出處理後的張量形狀
    print("Output tensor shape:", output_tensor.shape)

