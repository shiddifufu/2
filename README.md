# 1.代码部署
<img src="https://github.com/shiddifufu/2/blob/master/1.png" width="800" >

# 2.优化特征选择方法
<img src="https://github.com/shiddifufu/2/blob/master/2.png" width="800" >

# 3.样本平衡处理
<img src="https://github.com/shiddifufu/2/blob/master/3.png" width="800" >

# 4.增强模型评估指标
<img src="https://github.com/shiddifufu/2/blob/master/4.png" width="800" >

# 5.代码核心功能说明

## 多项式朴素贝叶斯分类器原理
1. 条件概率的独立性假设
多项式朴素贝叶斯基于以下核心假设：
特征条件独立性：所有特征（即文本中的词）在给定类别标签的条件下相互独立。
数学形式化表示为：$$P(X \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$
其中： $$X = \{ x_1, x_2, \dots, x_n \}$$ 表示特征向量（如词频）， y 表示类别标签（如垃圾邮件/正常邮件）。

2. 贝叶斯定理的应用
贝叶斯定理用于计算后验概率 P(y∣X)：
$$P(y \mid X) = \frac{P(y) \cdot P(X \mid y)}{P(X)}$$
在邮件分类中，具体应用形式为：
对每个类别 y （如 y=0 正常邮件， y=1 垃圾邮件），计算：
$$\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^n \log P(x_i \mid y) \right]$$
其中：
P(y) 是类别的先验概率（通过训练数据统计），
$$P(x_i \mid y)$$
是多项式分布下的条件概率（通过词频统计计算）。

3. 多项式分布的特点
输入特征：词频（非二元特征），即每个词在文档中的出现次数。

概率计算：
$$P(x_i \mid y) = \frac{N_{y,i} + \alpha}{N_y + \alpha \cdot n}$$
其中：
N_{y,i}：类别y中词x_i 的总出现次数，
N_y：类别y中所有词的总出现次数，
α：平滑系数（Laplace平滑，避免零概率问题）。

## 数据处理流程
1. 分词处理
实现逻辑：
将原始文本拆分为单词或词语序列。

英文：按空格和标点分割（如 nltk.word_tokenize）。

中文：需分词工具（如 jieba 库）。
如：
import jieba
text = "欢迎使用朴素贝叶斯分类器"
words = jieba.lcut(text)  # 输出：['欢迎', '使用', '朴素', '贝叶斯', '分类器']
2. 停用词过滤
实现逻辑：
移除无实际语义的高频词（如“的”、“是”、“the”、“and”）。
如：
stopwords = set(['的', '是', '和', '在', ...])
filtered_words = [word for word in words if word not in stopwords]
3. 其他预处理
小写转换：统一为小写（如 text.lower()）。

去除标点：正则表达式匹配删除（如 re.sub(r'[^\w\s]', '', text)）。

词干提取：还原单词到词根（如 nltk.PorterStemmer）。

## 特征构建过程
1. 高频词特征选择
数学表达：
选择训练集中词频最高的N个词作为特征，构建词频向量！！！！！！！！！
$$\mathbf{X}_{\mathrm{count}}^{(d)} = \left[ \mathrm{count}(w_1,d),\ \mathrm{count}(w_2,d),\ \ldots,\ \mathrm{count}(w_N,d) \right]$$
\mathrm{count}(w_1,d)：词w——1在文档d中的出现次数。

实现差异：
使用 CountVectorizer 统计词频，并限制最大特征数。
如：
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X_count = vectorizer.fit_transform(corpus)
2. TF-IDF特征加权
数学表达：
对词频进行加权，降低常见词权重，提升稀有词重要性：
$$\mathrm{TF\text{-}IDF}(w,d) = \mathrm{TF}(w,d) \times \log\left( \frac{N}{\mathrm{DF}(w) + 1} \right)$$
TF(w,d)：词w在文档d中的词频，
DF(w)：包含词w的文档数，
N：总文档数。

实现差异：
使用 TfidfVectorizer 自动计算 TF-IDF 值。
如：
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(corpus)
3. 对比分析
| 维度         | 高频词特征选择                    | TF-IDF特征加权                     |
|--------------|----------------------------------|-----------------------------------|
| **数学意义** | 仅反映词的出现频率                | 反映词在文档中的重要性             |
| **计算复杂度** | 低（只需统计词频）               | 高（需计算逆文档频率）             |
| **适用场景** | 简单分类任务                     | 需要区分关键词的复杂任务           |
| **代码实现** | `CountVectorizer`               | `TfidfVectorizer`                |
