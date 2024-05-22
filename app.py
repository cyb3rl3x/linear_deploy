import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Carregar o modelo salvo
model = joblib.load('modelo_regressao_linear.joblib')

# Carregar o dataset para visualização
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df = pd.read_csv(url, delim_whitespace=True, names=column_names)

# Selecionar as colunas de interesse (número de quartos e preço médio)
X_test = df[['RM']]
y_test = df['MEDV']

def main():
    st.title('Previsão de Preço de Casas Baseada no Número de Quartos')

    user_input = st.number_input('Digite o número de quartos', min_value=1.0, max_value=10.0, step=0.1)
    if st.button('Prever'):
        input_df = pd.DataFrame([user_input], columns=['RM'])
        prediction = model.predict(input_df)
        st.write(f'O preço estimado é: {prediction[0]:.2f} mil dólares')

        # Mostrar gráfico
        y_pred = model.predict(X_test)
        plt.figure()
        plt.scatter(df['RM'], df['MEDV'], color='blue', label='Dados')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regressão Linear')
        plt.xlabel('Número de Quartos')
        plt.ylabel('Preço Médio (em milhares de dólares)')
        plt.legend()
        plt.title('Gráfico de Regressão Linear')
        st.pyplot(plt)

if __name__ == '__main__':
    main()
