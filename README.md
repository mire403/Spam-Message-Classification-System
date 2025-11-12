## 项目名称
垃圾短信分类系统（SMS Spam Classifier）

## 项目概述
本系统旨在对中文短信进行自动分类，实现“正常短信”（ham）与“垃圾短信”（spam）之间的识别。系统采用经典机器学习流程：文本预处理 → 特征向量化（BoW / TF-IDF）→ 训练不同分类器（Multinomial Naive Bayes、Logistic Regression）→ 多维评估与可视化（混淆矩阵、ROC、词云），便于科研/课程/工程演示。

## 数据与预处理
- 数据格式：两类文本文件（`ham_data.txt`, `spam_data.txt`），每行一条短信；停用词表 `stop_words.utf8`。
- 预处理步骤：
  1. 去除空行与空白文本；
  2. 使用 `jieba` 进行中文分词；
  3. 正则与 `string.punctuation` 去除英文标点、特殊符号；
  4. 利用停用词表剔除常见无意义词（减少噪声）；
  5. 将分词后的 token 以空格连接（`"词1 词2 ..."`）以供 sklearn 向量器使用。

## 特征提取
- BoW（Bag-of-Words）：`sklearn.feature_extraction.text.CountVectorizer(min_df=1)`；
- TF-IDF：`sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True)`；
向量化后得到稀疏特征矩阵，分别用于训练不同模型。

## 模型与训练
- 划分策略：`train_test_split(test_size=0.3, random_state=42)`。
- 模型：
  - **Multinomial Naive Bayes（多项式朴素贝叶斯）**：适合词频数据，训练速度快，对短文本任务通常表现良好；
  - **Logistic Regression（逻辑回归）**：线性判别模型，配置为 `max_iter=1000` 以确保收敛，通常提供更好的泛化性能。
- 训练细节：分别在 BoW 与 TF-IDF 特征上训练朴素贝叶斯；在 TF-IDF 上训练逻辑回归以对比性能。

## 评估指标
- **准确率（Accuracy）**：整体预测正确率；
- **精确率（Precision）**：预测为垃圾短信中实际为垃圾的比例（关注误报）；
- **召回率（Recall）**：实际垃圾短信被正确识别的比例（关注漏报）；
- **F1 值**：Precision 与 Recall 的调和平均；
- **AUC / ROC**：通过 `predict_proba` 输出概率，绘制 ROC 曲线并计算 AUC 以衡量判别能力；
- **混淆矩阵**：可视化真阳/假阳/真阴/假阴的分布。

## 可视化产出
- 各模型的混淆矩阵（PNG 格式）；
- 模型训练/测试准确率对比柱状图；
- 各模型 ROC 曲线与 AUC 值；
- 分别基于训练集生成的正常短信与垃圾短信词云，直观展示高频词差异（用于解释模型行为）。

## 关键实现细节（来自代码）
- 停用词读取方式：逐行读取 `stop_words.utf8` 并在 `remove_stopwords` 中剔除（注意：确保 stopwords 行末无换行符影响比较）；
- 正则化：使用 `re` 与 `string.punctuation` 去除常见标点；
- 向量器参数示例：`TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True)`；
- 训练示例：`MultinomialNB().fit(X_train, y_train)` 与 `LogisticRegression(max_iter=1000).fit(X_train, y_train)`；
- ROC 与 AUC：通过 `model.predict_proba(X_test)[:, 1]` 获取正类概率，调用 `sklearn.metrics.roc_curve` 与 `auc`。

## 运行与复现实验
1. 准备数据并保存为 `data/ham_data.txt` 与 `data/spam_data.txt`，同时准备 `data/stop_words.utf8`；
2. 确保安装依赖：`numpy, jieba, matplotlib, seaborn, wordcloud, scikit-learn`；
3. 运行主脚本，程序会输出指标并保存图像文件，便于汇报与撰写实验结果。

## 优化建议（供后续扩展）
- 使用**更严格的停用词清洗**与**字符规范化**（如数字、手机号码、URL 特征抽取）提高召回/精确率；
- 尝试**n-gram（2-gram/3-gram）**或**字符级特征**以捕捉短文本中的模式；
- 引入**预训练语言模型（如 BERT / ERNIE）**或轻量级 Transformer（如 DistilBERT）做微调，可明显提升语义理解能力；
- 当数据不平衡时可使用采样（过采样/欠采样）或设置 class-weight 优化模型；
- 将模型封装为 REST API（Flask / FastAPI）以便部署到线上服务。
