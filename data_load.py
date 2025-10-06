import pandas as pd

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Файл успешно загружен")
        

        print(df.head())
        print(df.info())
        print(df.describe())
        print("Пропущенные значения по колонкам:")
        print(df.isnull().sum())

        return df
    except Exception as e:
        print(f"Ошибка при загрузке CSV: {e}")
        return None

df = load_csv("Salary_Data.csv")