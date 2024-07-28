import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from scipy.stats import mode
import pandas as pd


# 计算众数时更健壮的处理方法
def calculate_mode(row):
    m = mode(row)
    # 如果 mode 结果是数组形式，直接返回第一个元素
    if hasattr(m.mode, "__iter__") and not isinstance(m.mode, str):
        return m.mode[0]
    else:  # 否则直接返回 mode 结果
        return m.mode


#
# 需要ensemble的文件路径
file_paths = ['ensemble/submission_2024-02-28_19-09-27.csv',
              'ensemble/submission_2024-02-28_20-35-19.csv',
              'ensemble/submission_2024-02-29_12-04-36.csv', 'ensemble/submission_2024-02-29_12-04-42.csv',
              'ensemble/aug_submission_2024-02-29_20-46-46.csv',
              'ensemble/aug_submission_2024-02-29_22-30-45.csv',
              'ensemble/aug_submission_2024-02-29_22-34-59.csv',
              'ensemble/aug_submission_2024-02-29_22-34-59.csv',
              'ensemble/aug_submission_2024-02-29_23-00-12.csv',
              'ensemble/submission_2024-02-29_23-09-06.csv',
              'ensemble/submission_2024-02-29_23-09-06.csv',
              'ensemble/aug_submission_2024-02-29_23-11-23.csv'
              ]
# file_paths = ['ensemble/submission_2024-02-28_19-09-27.csv',
#               'ensemble/submission_2024-02-28_20-35-19.csv',
#               'ensemble/aug_submission_2024-02-29_20-54-38.csv', 'ensemble/submission_2024-02-29_12-04-42.csv',
#               'ensemble/aug_submission_2024-02-29_20-46-46.csv',
#               ]
# 读取所有文件
dfs = [pd.read_csv(path) for path in file_paths]

# 确保 'Id' 列匹配
for i in range(1, len(dfs)):
    assert dfs[0]['Id'].equals(dfs[i]['Id'])

# 合并所有DataFrame，以'Id'列为基准
merged_df = dfs[0][['Id']]
for i, df in enumerate(dfs):
    merged_df[f'Category_{i}'] = df['Category']

# 应用自定义的 calculate_mode 函数计算众数
mode_results = merged_df.drop(columns=['Id']).apply(calculate_mode, axis=1)

# 创建集成结果的 DataFrame
ensemble_df = pd.DataFrame({
    'Id': dfs[0]['Id'],
    'Category': mode_results
})

# 保存集成结果到新的 CSV 文件
ensemble_df.to_csv('ensemble_Category_mode.csv', index=False)
