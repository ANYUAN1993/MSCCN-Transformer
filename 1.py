import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import BorderlineSMOTE
from tensorflow.keras.layers import Input, Conv1D, Concatenate, MultiHeadAttention, LayerNormalization, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# 设置 matplotlib 字体
plt.rcParams['font.family'] = 'STZhongsong'  # 华文中宋
plt.rcParams['font.size'] = 7.5
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现性
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------
# 1. 数据加载与预处理 (增强版)
# --------------------------
def load_and_preprocess_data(data_path):
    """加载并预处理Linux审计日志数据"""
    # 加载所有CSV文件，指定混合类型列的dtype
    all_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    dfs = []
    
    # 定义混合类型列的dtype
    mixed_dtypes = {
        '_source.data.audit.syscall': 'str',
        '_source.data.audit.pid': 'str',
        '_source.data.audit.uid': 'str',
        '_source.data.audit.euid': 'str',
        '_source.data.audit.gid': 'str',
        '_source.data.audit.egid': 'str',
        '_source.data.audit.session': 'str',
        '_source.data.audit.exit': 'str',
        '_source.rule.level': 'str',
        '_source.data.audit.file.inode': 'str',
        '_source.data.audit.file.mode': 'str'
    }
    
    print(f"正在加载数据集: {data_path}")
    for file in all_files:
        try:
            print(f"处理文件: {file}")
            df = pd.read_csv(
                os.path.join(data_path, file),
                dtype=mixed_dtypes,
                low_memory=False,
                on_bad_lines='skip'
            )
            dfs.append(df)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    print(f"数据集加载完成，总样本数: {len(full_df)}")
    print(f"特征维度: {full_df.shape[1]}")
    
    # 关键特征选择
    features = [
        '_source.data.audit.syscall', '_source.data.audit.pid', '_source.data.audit.uid',
        '_source.data.audit.euid', '_source.data.audit.gid', '_source.data.audit.egid',
        '_source.data.audit.session', '_source.data.audit.exit', '_source.data.audit.success',
        '_source.rule.level', '_source.data.audit.file.inode', '_source.data.audit.file.mode',
        '_source.data.audit.command', '_source.data.audit.exe'
    ]
    
    # MITRE攻击类型标签
    mitre_labels = full_df['_source.rule.mitre.technique'].fillna('Normal')
    
    # 特征工程
    processed_df = full_df[features].copy()
    
    # 处理缺失值
    processed_df.fillna(0, inplace=True)
    
    # 转换布尔值为数值
    processed_df['_source.data.audit.success'] = processed_df['_source.data.audit.success'].map(
        {'yes': 1, 'no': 0})
    
    # 清理数值列中的非数值字符
    numeric_cols = [
        '_source.data.audit.syscall', '_source.data.audit.pid', '_source.data.audit.uid',
        '_source.data.audit.euid', '_source.data.audit.gid', '_source.data.audit.egid',
        '_source.data.audit.session', '_source.data.audit.exit', '_source.rule.level',
        '_source.data.audit.file.inode'
    ]
    
    for col in numeric_cols:
        # 移除空格和非数字字符
        processed_df[col] = processed_df[col].astype(str).str.replace(r'\s+', '', regex=True)
        # 将空字符串转换为0
        processed_df[col] = processed_df[col].replace('', '0')
        # 转换为数值类型
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
    
    # 特殊处理file.mode列
    processed_df['_source.data.audit.file.mode'] = processed_df['_source.data.audit.file.mode'].astype(str)
    processed_df['file_mode_numeric'] = processed_df['_source.data.audit.file.mode'].apply(
        lambda x: int(x, 8) if isinstance(x, str) and x.strip() != '' and x.strip().isdigit() else 0
    )
    
    # 标签编码命令和可执行文件
    le_command = LabelEncoder()
    le_exe = LabelEncoder()
    
    # 处理命令列中的缺失值
    processed_df['_source.data.audit.command'] = processed_df['_source.data.audit.command'].fillna('unknown_command')
    processed_df['command_encoded'] = le_command.fit_transform(
        processed_df['_source.data.audit.command'].astype(str))
    
    # 处理可执行文件列中的缺失值
    processed_df['_source.data.audit.exe'] = processed_df['_source.data.audit.exe'].fillna('unknown_exe')
    processed_df['exe_encoded'] = le_exe.fit_transform(
        processed_df['_source.data.audit.exe'].astype(str))
    
    # 删除原始分类列
    processed_df.drop([
        '_source.data.audit.command', 
        '_source.data.audit.exe',
        '_source.data.audit.file.mode'
    ], axis=1, inplace=True)
    
    # 提取数值特征
    numerical_features = processed_df.columns.tolist()
    
    # 检查并处理NaN值
    print("处理前NaN值数量:", processed_df.isna().sum().sum())
    processed_df.fillna(0, inplace=True)
    print("处理后NaN值数量:", processed_df.isna().sum().sum())
    
    # 标准化数值特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(processed_df[numerical_features])
    
    # 检查标准化后的NaN值
    nan_count = np.isnan(scaled_features).sum()
    print(f"标准化后NaN值数量: {nan_count}")
    if nan_count > 0:
        # 使用中位数填充NaN值
        imputer = SimpleImputer(strategy='median')
        scaled_features = imputer.fit_transform(scaled_features)
    
    # 标签编码
    le_mitre = LabelEncoder()
    y = le_mitre.fit_transform(mitre_labels)
    
    return scaled_features, y, le_mitre, scaler, le_command, le_exe

# --------------------------
# 2. 时间序列构建 (增强版)
# --------------------------
def create_sequences(X, y, window_size=50, step_size=10):
    """将数据转换为时间序列格式"""
    # 检查数据量是否足够
    if len(X) < window_size:
        raise ValueError(f"数据量({len(X)})小于窗口大小({window_size})，无法创建序列")
    
    X_seq = []
    y_seq = []
    
    # 检查并处理输入数据中的NaN值
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        print(f"发现 {nan_mask.sum()} 个包含NaN值的样本，正在处理...")
        # 使用列中位数填充NaN值
        col_medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[0]):
            if nan_mask[i]:
                nan_indices = np.isnan(X[i])
                X[i][nan_indices] = col_medians[nan_indices]
    
    # 创建序列
    print(f"正在创建时间序列 (窗口大小={window_size}, 步长={step_size})...")
    for i in range(0, len(X) - window_size, step_size):
        seq = X[i:i+window_size]
        # 再次检查序列中的NaN值
        if np.isnan(seq).any():
            # 使用序列中位数填充NaN值
            seq_median = np.nanmedian(seq, axis=0)
            nan_indices = np.isnan(seq)
            seq[nan_indices] = np.take(seq_median, np.where(nan_indices)[1])
        X_seq.append(seq)
        # 使用窗口最后一个样本的标签
        y_seq.append(y[i+window_size-1])  
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # 最终检查
    nan_count = np.isnan(X_seq).sum()
    if nan_count > 0:
        print(f"序列中仍有 {nan_count} 个NaN值，使用0填充")
        X_seq = np.nan_to_num(X_seq, nan=0.0)
    
    return X_seq, y_seq

# --------------------------
# 3. 处理数据不平衡 (增强版) - 修复版
# --------------------------
def balance_dataset(X, y):
    """使用Borderline-SMOTE处理数据不平衡，过滤样本不足的类别"""
    # 检查输入数据中的NaN值
    nan_count_x = np.isnan(X).sum()
    nan_count_y = np.isnan(y).sum()
    print(f"平衡前 - X中NaN数量: {nan_count_x}, y中NaN数量: {nan_count_y}")
    
    # 处理NaN值
    if nan_count_x > 0:
        print("X中存在NaN值，使用0填充")
        X = np.nan_to_num(X, nan=0.0)
    
    if nan_count_y > 0:
        print("y中存在NaN值，使用0填充")
        y = np.nan_to_num(y, nan=0.0)
    
    # 重塑数据以适应SMOTE (samples, features)
    original_shape = X.shape
    X_flat = X.reshape(X.shape[0], -1)
    
    # 再次检查NaN值
    if np.isnan(X_flat).any():
        print("X_flat中存在NaN值，使用0填充")
        X_flat = np.nan_to_num(X_flat, nan=0.0)
    
    # 确保y是整数类型
    y = y.astype(int)
    
    # 统计每个类别的样本数
    class_counts = Counter(y)
    print("原始类别分布:")
    for cls, count in class_counts.items():
        print(f"类别 {cls}: {count} 个样本")
    
    # 过滤掉样本数少于6的类别（因为SMOTE需要至少6个样本）
    min_samples = 6
    valid_indices = [i for i, label in enumerate(y) if class_counts[label] >= min_samples]
    invalid_indices = [i for i, label in enumerate(y) if class_counts[label] < min_samples]
    
    print(f"过滤掉 {len(invalid_indices)} 个样本（属于样本不足的类别）")
    
    if len(valid_indices) == 0:
        raise ValueError("没有足够样本的类别可用于过采样")
    
    X_valid = X_flat[valid_indices]
    y_valid = y[valid_indices]
    
    # 处理数据不平衡
    print("应用Borderline-SMOTE处理数据不平衡...")
    smote = BorderlineSMOTE(random_state=42, k_neighbors=5)  # 明确设置k_neighbors
    X_res, y_res = smote.fit_resample(X_valid, y_valid)
    
    # 恢复为序列形状
    X_res = X_res.reshape(-1, original_shape[1], original_shape[2])
    
    # 添加被过滤掉的样本（如果存在）
    if len(invalid_indices) > 0:
        X_invalid = X_flat[invalid_indices].reshape(-1, original_shape[1], original_shape[2])
        y_invalid = y[invalid_indices]
        
        X_res = np.concatenate([X_res, X_invalid], axis=0)
        y_res = np.concatenate([y_res, y_invalid], axis=0)
    
    print(f"重采样后数据集形状: {X_res.shape}")
    
    # 统计重采样后的类别分布
    new_class_counts = Counter(y_res)
    print("重采样后类别分布:")
    for cls, count in new_class_counts.items():
        print(f"类别 {cls}: {count} 个样本")
    
    return X_res, y_res

# --------------------------
# 4. 双分支Transformer-MSCNN模型
# --------------------------
def build_transformer_mscnn_model(input_shape, num_classes):
    """构建双分支Transformer-MSCNN模型"""
    inputs = Input(shape=input_shape)
    
    # MSCNN分支 - 多尺度卷积
    conv1x1 = Conv1D(32, 1, padding='same', activation='relu')(inputs)
    conv3x1 = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    conv5x1 = Conv1D(32, 5, padding='same', activation='relu')(inputs)
    conv7x1 = Conv1D(32, 7, padding='same', activation='relu')(inputs)
    
    concat_conv = Concatenate(axis=-1)([conv1x1, conv3x1, conv5x1, conv7x1])
    conv_out = Conv1D(64, 3, activation='relu')(concat_conv)
    mscnn_out = GlobalMaxPooling1D()(conv_out)
    
    # Transformer分支
    # 嵌入层
    x = Dense(128)(inputs)
    
    # Transformer编码器块
    for _ in range(2):
        # 多头注意力
        attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # 前馈网络
        ffn = Dense(512, activation='relu')(x)
        ffn = Dense(128)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    transformer_out = GlobalAveragePooling1D()(x)
    
    # 特征融合
    combined = Concatenate()([mscnn_out, transformer_out])
    
    # 分类头
    dense = Dense(128, activation='relu')(combined)
    outputs = Dense(num_classes, activation='softmax')(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# --------------------------
# 5. 模型训练与评估 (优化版)
# --------------------------
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, num_classes):
    """训练和评估模型"""
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # 构建模型
    print("构建双分支Transformer-MSCNN模型...")
    model = build_transformer_mscnn_model(input_shape, num_classes)
    model.summary()
    
    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),  # 减少耐心值
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)  # 减少耐心值
    ]
    
    # 训练模型 (增加到50轮)
    print("开始训练模型 (50轮)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # 增加到50轮
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估模型
    print("评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}")
    
    # 预测
    print("生成预测结果...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    return model, history, y_pred_classes, test_acc

# --------------------------
# 6. 可视化与分析 (修复版)
# --------------------------
def visualize_results(history, X_test, y_test, y_pred, unique_labels, test_acc):
    """可视化训练过程和评估结果"""
    # 训练历史：准确率和损失
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='训练准确率', linestyle='-', linewidth=0.3)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='验证准确率', linestyle='--', linewidth=0.3)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='训练损失', linestyle='-.', linewidth=0.3)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='验证损失', linestyle=':', linewidth=0.3)
   # plt.title('模型准确率和损失')
    plt.ylabel('准确率/损失')
    plt.xlabel('轮数/次')
    plt.legend(edgecolor='black')
    plt.grid(True, linestyle=':', linewidth=0.2)
    plt.tight_layout()
    plt.savefig('training_history_accuracy_loss.svg')
    plt.show()
    
    # 准确率、精确率、召回率和F1分数
    accuracy = test_acc
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [accuracy, precision, recall, f1]
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, width=0.4, linewidth=0.2)
    for i, v in enumerate(values):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    #plt.title('模型评估指标')
    plt.ylabel('值')
    plt.xlabel('指标')
    plt.grid(True, linestyle=':', linewidth=0.2)
    plt.tight_layout()
    plt.savefig('metrics.svg')
    plt.show()
    
    # 获取实际存在的类别
    actual_labels = np.unique(np.concatenate([y_test, y_pred]))
    num_classes = len(actual_labels)
    
    # 生成类别名称
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=actual_labels)
    
    plt.figure(figsize=(max(10, num_classes*0.8), max(8, num_classes*0.6)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, 
               yticklabels=class_names, linewidths=0.2)
    #plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.xticks(rotation=45)
    plt.savefig('confusion_matrix.svg')
    plt.show()
    
    # 关键特征分析
    plt.figure(figsize=(10, 6))
    # 使用第一个特征作为示例
    feature_idx = 0
    
    # 检查是否有正常流量和攻击流量
    if len(np.unique(y_test)) > 1:
        # 找到正常类别的索引（假设0是正常类别）
        normal_class = 0
        attack_indices = np.where(y_test != normal_class)[0][:100]
        normal_indices = np.where(y_test == normal_class)[0][:100]
        
        if len(attack_indices) > 0 and len(normal_indices) > 0:
            plt.plot(X_test[attack_indices, -1, feature_idx], label='攻击流量', linestyle='-', linewidth=0.3)
            plt.plot(X_test[normal_indices, -1, feature_idx], label='正常流量', linestyle='--', linewidth=0.3)
            #plt.title('系统调用特征对比')
            plt.xlabel('时间窗口/个')
            plt.ylabel('标准化系统调用值')
            plt.legend(edgecolor='black')
            plt.grid(True, linestyle=':', linewidth=0.2)
            plt.savefig('feature_comparison.svg')
            plt.show()
        else:
            print("警告: 无法找到足够的正常或攻击流量样本进行对比")
    else:
        print("警告: 数据集中只有一个类别，无法进行正常/攻击流量对比")

# --------------------------
# 7. 消融实验 (优化版) - 修复串行架构
# --------------------------
def ablation_study(X_train, y_train, X_val, y_val, X_test, y_test, num_classes):
    """执行消融实验"""
    # 确保标签连续
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    y_train = np.vectorize(label_map.get)(y_train)
    y_val = np.vectorize(label_map.get)(y_val)
    y_test = np.vectorize(label_map.get)(y_test)
    actual_num_classes = len(unique_labels)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    results = {}
    
    print("开始消融实验 (10轮/模型)...")
    
    # 仅MSCNN模型
    print("训练仅MSCNN模型...")
    inputs = Input(shape=input_shape)
    conv1x1 = Conv1D(32, 1, padding='same', activation='relu')(inputs)
    conv3x1 = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    conv5x1 = Conv1D(32, 5, padding='same', activation='relu')(inputs)
    conv7x1 = Conv1D(32, 7, padding='same', activation='relu')(inputs)
    concat_conv = Concatenate(axis=-1)([conv1x1, conv3x1, conv5x1, conv7x1])
    conv_out = Conv1D(64, 3, activation='relu')(concat_conv)
    mscnn_out = GlobalMaxPooling1D()(conv_out)
    outputs = Dense(actual_num_classes, activation='softmax')(mscnn_out)
    mscnn_model = Model(inputs=inputs, outputs=outputs)
    mscnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mscnn_model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)  # 降低至10轮
    _, mscnn_acc = mscnn_model.evaluate(X_test, y_test, verbose=0)
    results['仅MSCNN'] = mscnn_acc
    
    # 仅Transformer模型
    print("训练仅Transformer模型...")
    inputs = Input(shape=input_shape)
    x = Dense(128)(inputs)
    for _ in range(2):
        attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn = Dense(512, activation='relu')(x)
        ffn = Dense(128)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    transformer_out = GlobalAveragePooling1D()(x)
    outputs = Dense(actual_num_classes, activation='softmax')(transformer_out)
    transformer_model = Model(inputs=inputs, outputs=outputs)
    transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    transformer_model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)  # 降低至10轮
    _, transformer_acc = transformer_model.evaluate(X_test, y_test, verbose=0)
    results['仅Transformer'] = transformer_acc
    
    # 串行架构 - 修复版
    print("训练串行架构模型...")
    inputs = Input(shape=input_shape)
    conv1x1 = Conv1D(32, 1, padding='same', activation='relu')(inputs)
    conv3x1 = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    conv5x1 = Conv1D(32, 5, padding='same', activation='relu')(inputs)
    conv7x1 = Conv1D(32, 7, padding='same', activation='relu')(inputs)
    concat_conv = Concatenate(axis=-1)([conv1x1, conv3x1, conv5x1, conv7x1])
    conv_out = Conv1D(64, 3, activation='relu')(concat_conv)
    conv_pool = GlobalMaxPooling1D()(conv_out)
    
    # 修复: 使用RepeatVector层替代tf.repeat
    repeated = RepeatVector(input_shape[0])(conv_pool)
    
    x = Dense(128)(repeated)
    for _ in range(2):
        attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn = Dense(512, activation='relu')(x)
        ffn = Dense(128)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    transformer_out = GlobalAveragePooling1D()(x)
    outputs = Dense(actual_num_classes, activation='softmax')(transformer_out)
    serial_model = Model(inputs=inputs, outputs=outputs)
    serial_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    serial_model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)  # 降低至10轮
    _, serial_acc = serial_model.evaluate(X_test, y_test, verbose=0)
    results['串行架构'] = serial_acc
    
    return results

# --------------------------
# 主执行流程 (优化版)
# --------------------------
def main():
    # 数据集路径
    DATA_PATH = r"C:\Users\Administrator\Desktop\data1\zuizhong\1"
    
    # 1. 加载和预处理数据
    print("正在加载和预处理数据...")
    X, y, le_mitre, scaler, le_command, le_exe = load_and_preprocess_data(DATA_PATH)
    
    # 检查NaN值
    nan_count = np.isnan(X).sum()
    print(f"预处理后X中NaN值数量: {nan_count}")
    if nan_count > 0:
        print("使用中位数填充X中的NaN值")
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    # 2. 构建时间序列
    print("构建时间序列数据...")
    try:
        X_seq, y_seq = create_sequences(X, y, window_size=50, step_size=10)
        print(f"时间序列形状: {X_seq.shape}, 标签形状: {y_seq.shape}")
    except ValueError as e:
        print(f"创建时间序列时出错: {e}")
        return
    
    # 3. 处理数据不平衡
    print("处理数据不平衡...")
    try:
        X_balanced, y_balanced = balance_dataset(X_seq, y_seq)
    except ValueError as e:
        print(f"平衡数据集时出错: {e}")
        return
    
    # 标签重映射
    print("执行标签重映射...")
    unique_labels = np.unique(y_balanced)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y_balanced = np.vectorize(label_map.get)(y_balanced)
    num_classes = len(unique_labels)
    
    # 输出标签信息
    print(f"重映射后唯一标签值: {unique_labels}")
    print(f"标签最大值: {np.max(y_balanced)}")
    print(f"实际类别数: {num_classes}")
    
    # 4. 数据集划分
    print("划分训练集、验证集和测试集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 最终NaN检查
    print("最终数据集NaN检查:")
    print(f"训练集X: {np.isnan(X_train).sum()}, y: {np.isnan(y_train).sum()}")
    print(f"验证集X: {np.isnan(X_val).sum()}, y: {np.isnan(y_val).sum()}")
    print(f"测试集X: {np.isnan(X_test).sum()}, y: {np.isnan(y_test).sum()}")
    
    # 5. 训练和评估主模型
    print("训练双分支Transformer-MSCNN模型...")
    model, history, y_pred, test_acc = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test, num_classes
    )
    
    # 6. 可视化结果
    print("生成可视化结果...")
    visualize_results(history, X_test, y_test, y_pred, unique_labels, test_acc)
    
    # 7. 消融实验
    print("进行消融实验...")
    ablation_results = ablation_study(X_train, y_train, X_val, y_val, X_test, y_test, num_classes)
    
    # 输出消融实验结果
    print("\n消融实验结果:")
    for model_type, acc in ablation_results.items():
        print(f"{model_type}: 准确率 = {acc:.4f}")
    
    # 绘制消融实验准确率柱状图
    models = list(ablation_results.keys())
    accuracies = list(ablation_results.values())
    models.append('本文模型')
    accuracies.append(test_acc)
    
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, width=0.4, linewidth=0.2)
    for i, v in enumerate(accuracies):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
   # plt.title('消融实验模型准确率对比')
    plt.ylabel('准确率')
    plt.xlabel('模型类型')
    plt.grid(True, linestyle=':', linewidth=0.2)
    plt.tight_layout()
    plt.savefig('ablation_study_accuracies.svg')
    plt.show()
    
    # 保存模型
    model.save('transformer_mscnn_apt_detection.h5')
    print("模型已保存为 'transformer_mscnn_apt_detection.h5'")

if __name__ == "__main__":
    main()