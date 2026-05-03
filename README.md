一、AI交互日志
M1: 数据处理与特征工程
关键 Prompt：
"我正在处理300万行的nyc taxi data Parquet数据。请帮我清洗异常值，并生成行程耗时与速度两个衍生特征。"
AI输出摘要：
1. 车费金额必须>0
2. 行程距离必须>0且<1000，距离为0的订单没有分析价值，过长的距离可能是数据错误
3. 乘客人数过滤掉0人和>9人的异常记录。
4. 行程时间: 剔除接单时间晚于下车时间的错误数据，以及异常漫长的订单的数据；
5. 生成了我说的两个特征。
这里的数据筛选比较死板简单，ai输出我看过没问题就全部采用了。

M2: 分析可视化模块
Prompt：
"请帮我用seaborn画一张图，展示车费与行程距离的关系，保存到outputs文件夹下。"
ai生成了一段使用 `sns.scatterplot(x='trip_distance', y='total_amount', data=df)` 的代码。
AI错误记录：
直接对 300 万个散点进行 `scatterplot '。运行时，程序在M2模块卡死，VS Code终端无响应，最终放弃。
如何修正：我意识到不能让AI盲目处理大数据。我向AI发修改指令：“数据量太大请修改逻辑：先对数据进行处理再绘制散点图”。AI 随后更新了代码，随机选择数据后成功生成，且清晰展示了距离与车费的正相关性。

M3: 预测模型构建
Prompt：
"用PyTorch写一个神经网络模型，8：2划分训练集与验证集，预测出行需求量。要求输出Loss曲线和MSE,RMSE。"
ai输出摘要：提供了一个包含三个全连接层的 `nn.Sequential` 网络，并给出了完整的训练循环。（后面有该部分分析对比）

M4: 智能问答系统接口
关键Prompt：
"我要实现这个交通系统的用户问答循环，并打算用大模型API兜底系统没有处理的问题。"
ai错误纠正记录：
ai一开始直接让我把自己的apikey复制到main。py里面，但我了解到这样子风险很大，人人都可以使用我的token额度了。于是我修改成我在终端复制，提升了安全性。
还有与ai交流中我想到用户可能乱输入问题，于是我在prompt里加入：如果用户问了和交通、数据科学、纽约市毫无关系的问题，请委婉拒绝，并引导他们询问数据相关的问题

二、 三阶段对比M3 PyTorch预测模型
阶段一：Native版
我查阅了PyTorch资料和书本的实例，试图独立完成数据的Tensor转换和前向传播。
代码片段：
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    
    model = nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        preds = model(X_train_tensor)
        loss = nn.MSELoss()(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
总结：效率极低。耗费了一个小时理解张量维度对齐问题，只是复现代码，短时间内没办法深度理解，拟合效果极差，完全没有体现出深度学习的优势。

阶段二：Prompt版
我将需求描述给AI，AI给出了一个过关的深度神经网络模板。
代码片段：
    class DemandPredictor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # 试图把几十万行数据一次性塞进模型算梯度
    outputs = model(X_train_tensor) 
总结：代码很快生成，语法规范。当我运行这段代码时，由于试图将庞大的数据一次性进行矩阵乘法，导致程序崩溃。此时我理解到ai懂语法，但在不知道整体项目时很“暴力”，不懂得变通适应，可能要自己调整或者给ai更多细节。

阶段三：Vibe版
基于前两次的失败，我开始与AI互动探讨：“训练的准确率如何提高？”、“作业要求和随机森林对比，谁会赢？”
最终代码片段（即现存项目中的正式代码）：
    # 避免数据量过大,选取单量最活跃的Top 20区域进行训练预测
    top20_locs = agg_df.groupby('PULocationID')['demand'].sum().nlargest(20).index
    model_df = agg_df[agg_df['PULocationID'].isin(top20_locs)].copy()
    # 提取特征列和目标列
    feature_cols = ['PULocationID', 'pickup_hour', 'pickup_weekday', 'is_weekend', 'is_peak_hour']
    X = model_df[feature_cols].values
    y = model_df['demand'].values
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test）
    # PyTorch神经网络训练
    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    # 初始化模型、损失函数(MSE)和优化器(Adam)
    input_dim = X_train.shape[1]
    nn_model = DemandPredictorNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
    epochs = 200
    train_losses = []
    for epoch in range(epochs):
        # 前向传播
        outputs = nn_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor) 
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if (epoch+1) % 50 == 0:
            print(f"     Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
  总结：这是人机协作的最高阶段。我负责掌控工程全局约束与需求设计，AI负责填补算法细节。我不仅高效完成了代码，更从 AI 那里学到了RF与神经网络的优劣势对比。

三、反思
做这个大作业之前，我以为有AI加持两三个小时就能搞定。但在历经了绘图乱码、变量名混乱、API报错后，我找到了真正使用它的正确方法。  
刚开始做M1清洗和M3神经网络时，我描述这个模块的功能，直接复制AI给的代码运行。但处理Parquet数据时，AI给的PyTorch训练代码试图把全部数据一次性塞进张量，直接导致我的电脑卡死。我明白了AI只负责给出理论上的代码。最后是我让AI “只取Top 20区域进行”，模型才跑通。 
我还发现如果把对话拉得太长，AI就会开始会忘记我之前在M1中已经定义好的字段名，导致后面生成的代码跑不起来报KeyError。为了解决这个问题，我不得不自己在数据官网找回字符文档，理清所有特征的维度和变量名，在提问时不时加上上下文和数据文档。
在搞M4的大模型选做题时，我本来以为拿个免费API Key填进去就行了，但依旧经历了模型调用错误。在与ai的聊天中我也做了改进，我没有写复杂的if-else去判断用户是不是在瞎聊天，在System Prompt里加了一句严厉的约束语“遇到非交通问题必须拒绝”，系统就变聪明了。  
总结：AI不是来替我写完作业的，它更像是一个记忆力好但没有常识的实习生。有它辅助，我不用经常查复杂的文档了，但系统要实现什么功能，有什么风险依然只能由我自己来把握。
