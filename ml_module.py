import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):

    try:
        df = pd.read_csv(file_path)
        print(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} колонок")
        return df
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

def train_and_plot(df, model_type="linear"):

    #Обучение модели для предсказания зарплаты по опыту работы

    X = df[['YearsExperience']]
    y = df['Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == "forest":
        model = RandomForestRegressor(random_state=42, n_estimators=100)
    else:
        return None

    model.fit(X_train, y_train)
    
    # Делаем предсказания
    y_pred = model.predict(X_test)

    # Оцениваем качество
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_type.upper()} модель:")
    print(f"Ошибка (MSE): {mse:.2f}")
    print(f"Точность (R²): {r2:.2f}")

    # Простая визуализация
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Настоящая зарплата")
    plt.ylabel("Предсказанная зарплата")
    plt.title(f"Предсказания {model_type} модели")
    plt.show()

    return model

def predict_salary(model, years_experience):
    """Предсказание зарплаты для заданного опыта работы"""
    prediction = model.predict([[years_experience]])[0]
    return prediction

if __name__ == "__main__":
    # Загружаем данные
    df = load_data("Salary_Data.csv")
    
    if df is not None:
        # Обучаем модели
        print("Обучаем модели машинного обучения...")
        
        linear_model = train_and_plot(df, "linear")
        tree_model = train_and_plot(df, "tree") 
        forest_model = train_and_plot(df, "forest")
        
    else:
        print("Не удалось загрузить данные")