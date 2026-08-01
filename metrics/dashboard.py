import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import os

st.set_page_config(page_title="RL Betting Dashboard", layout="wide")

def get_data():
    db_path = "data/history.db"
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    try:
        history = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
        predictions = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    except:
        history, predictions = pd.DataFrame(), pd.DataFrame()
    conn.close()
    return history, predictions

st.title("📊 Painel de Métricas - Reinforcement Learning")

placeholder = st.empty()

while True:
    history, predictions = get_data()
    
    with placeholder.container():
        if predictions.empty:
            st.warning("Aguardando dados iniciais...")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total = len(predictions)
            acc_overall = predictions['is_correct'].mean() * 100
            
            last_100 = predictions.head(100)
            acc_100 = last_100['is_correct'].mean() * 100
            
            last_500 = predictions.head(500)
            acc_500 = last_500['is_correct'].mean() * 100
            
            col1.metric("Precisão Geral", f"{acc_overall:.2f}%")
            col2.metric("Últimos 100", f"{acc_100:.2f}%")
            col3.metric("Últimos 500", f"{acc_500:.2f}%")
            col4.metric("Total de Jogos", total)
            
            # Graphs
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Evolução da Precisão (Média Móvel)")
                predictions['rolling_acc'] = predictions['is_correct'].iloc[::-1].rolling(window=50).mean().iloc[::-1]
                fig_acc = px.line(predictions, x=predictions.index, y='rolling_acc', title="Precisão Móvel (Janela 50)")
                st.plotly_chart(fig_acc, use_container_width=True)
                
            with c2:
                st.subheader("Matriz de Confusão")
                if len(predictions) > 1:
                    y_true = predictions['actual_color']
                    y_pred = predictions['predicted_color']
                    labels = sorted(list(set(y_true) | set(y_pred)))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, title="Matriz de Confusão")
                    st.plotly_chart(fig_cm, use_container_width=True)
            
            st.subheader("Histórico Recente")
            st.dataframe(predictions.head(20), use_container_width=True)

    time.sleep(10)
