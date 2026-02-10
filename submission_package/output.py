import pandas as pd
from pathlib import Path
from typing import List


class DataOutput:
    def __init__(self, results: List[List], output_file: str = "./submission.xlsx"):
        self.results = results
        self.output_file = output_file


    def save_submission(self) -> None:
        """
        保存提交文件为xlsx格式

        参数:
            data: 二维列表，每行格式 [user_id(str), category(str), start(int64), end(int64)]
                  时间戳是13位，需要astype（“int64”）
                  例: [["HNU22001", "A", 1000000000000, 2000000000000]
            output: 输出文件路径

        数据类型:
            user_id: str
            category: str
            start: int64
            end: int64
        """
        # 检查数据格式
        if not self.results:
            raise ValueError("data 不能为空")

        if not all(len(row) == 4 for row in self.results):
            raise ValueError("每行必须包含 4 个元素 [user_id, category, start, end]")

        # 创建 DataFrame
        df = pd.DataFrame(self.results, columns=["user_id", "category", "start", "end"])

        # 强制类型转换
        df["user_id"] = df["user_id"].astype(str)
        df["category"] = df["category"].astype(str)
        df["start"] = df["start"].astype("int64")
        df["end"] = df["end"].astype("int64")

        # 保存为 xlsx
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.output_file, index=False, engine='openpyxl')

        print(f"✓ 已保存 {len(df)} 条记录到 {self.output_file}")


# 使用示例
if __name__ == "__main__":
    # 方式1: 直接传二维列表


    data = [
        ["HNU22001", "A", 10, 1111111111112],
        ["HNU22002", "B", 5, 15],
        ["HNU22003", "C", 0, 30]
    ]
    dataoutput = DataOutput(data)
    dataoutput.save_submission()
    # save_submission(data)

