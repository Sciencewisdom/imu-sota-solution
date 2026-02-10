from pathlib import Path
from typing import Dict


class DataReader:
    def __init__(self, dst: str):
        self.dst = dst

    def read_data(self) -> Dict[str, str]:
        """
        读取指定目录下所有 txt 文件，返回 {文件ID: 文本内容}
        """
        out = Path(self.dst)
        result = {}

        if not out.exists():
            print(f"目录不存在: {out}")
            return result

        # 读取所有txt文件
        for txt_file in sorted(out.glob("*.txt")):
            file_id = txt_file.stem  # 文件名（不含扩展名）
            try:
                text = txt_file.read_text(encoding='utf-8')
                result[file_id] = text
            except Exception as e:
                print(f"读取 {txt_file} 失败: {e}")

        return result




if __name__ == "__main__":
    # 读取数据
    reader = DataReader("./test_data")
    data = reader.read_data()
    print(data)