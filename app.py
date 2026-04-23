import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import TeamSummary, CommonTeamRoster, PlayerCareerStats
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Страница 1: Конфигурация ---
st.set_page_config(page_title="NBA Analytics Hub", layout="wide")
st.title("🏀 NBA Analytics: Статистика, Прогнозы и Вероятность")

# Боковая панель поиска
st.sidebar.header("🔍 Поиск команды")
all_teams = teams.get_teams()
team_names = {team['full_name']: team['id'] for team in all_teams}
selected_team_name = st.sidebar.selectbox("Выберите команду", team_names.keys())

if selected_team_name:
    team_id = team_names[selected_team_name]
    
    # --- 1. Загрузка данных о команде ---
    with st.spinner("Загрузка данных..."):
        try:
            # Общая статистика команды
            team_stats = TeamSummary(team_id=team_id).get_data_frames()[0]
            # Ростер (игроки)
            roster = CommonTeamRoster(team_id=team_id).get_data_frames()[0]
        except Exception as e:
            st.error(f"Ошибка API: {e}. NBA обновляет заголовки запросов.")
            st.stop()

    # --- Интерфейс ---
    st.header(f"📊 {selected_team_name}")
    
    # Отображение ключевых метрик команды
    col1, col2, col3, col4 = st.columns(4)
    if not team_stats.empty:
        col1.metric("🏆 Победы", team_stats['WINS'].iloc[0])
        col2.metric("📉 Поражения", team_stats['LOSSES'].iloc[0])
        col3.metric("⛹️ Win %", f"{float(team_stats['WIN_PCT'].iloc[0]):.1%}")
        col4.metric("📈 Забивают за игру", round(team_stats['PTS'].iloc[0] / team_stats['GP'].iloc[0], 1))

    st.subheader("👥 Состав команды и персональные прогнозы")
    
    # --- 2. Аналитика по каждому игроку ---
    player_data = []
    for _, player in roster.iterrows():
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER']
        
        # Получаем карьерную статистику (последние 2 сезона)
        try:
            career = PlayerCareerStats(player_id=player_id).get_data_frames()[0]
            if len(career) >= 2:
                last_year = career.iloc[-1]
                prev_year = career.iloc[-2]
                
                # --- 3. Модель прогноза (ИИ) ---
                # Данные за 2 года: [Minutes, Points, Rebounds, Assists]
                X_train = [[prev_year['MIN'], prev_year['PTS'], prev_year['REB'], prev_year['AST']],
                           [last_year['MIN'], last_year['PTS'], last_year['REB'], last_year['AST']]]
                y_train = [prev_year['PTS'], last_year['PTS']]
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Прогноз на основе усредненных показателей (простое предположение)
                X_pred = [[last_year['MIN'] * 1.02, last_year['PTS'], last_year['REB'] * 1.01, last_year['AST']]]
                predicted_pts = round(model.predict(X_pred)[0], 1)
                
                # Простая эвристика вероятности (чем стабильнее, тем выше шанс)
                volatility = abs(last_year['PTS'] - prev_year['PTS']) / max(prev_year['PTS'], 1)
                probability = max(0, min(100, round(100 - (volatility * 100), 0))) # Шкала 0-100%
                
                player_data.append({
                    "Имя": player_name,
                    "Позиция": player['POSITION'],
                    "PPG 2024": round(last_year['PTS'], 1),
                    "Прогноз PPG": predicted_pts,
                    "Вероятность (%)": probability,
                    "Подборы": last_year['REB'],
                    "Передачи": last_year['AST']
                })
        except:
            player_data.append({"Имя": player_name, "Позиция": player['POSITION'], "Статус": "Недостаточно данных"})

    # Отображение таблицы игроков
    df_players = pd.DataFrame(player_data)
    if not df_players.empty and 'PPG 2024' in df_players.columns:
        st.dataframe(df_players, use_container_width=True)
        
        # График: Текущее vs Прогноз
        fig = px.bar(df_players, x='Имя', y=['PPG 2024', 'Прогноз PPG'], 
                     title="Сравнение текущей результативности с ИИ-прогнозом",
                     barmode='group', color_discrete_map={'PPG 2024': '#1f77b4', 'Прогноз PPG': '#ff7f0e'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Метрика вероятности успеха команды (средняя стабильность)
        avg_prob = df_players['Вероятность (%)'].mean()
        st.metric("🎲 Общая вероятность стабильности команды", f"{round(avg_prob)}%")
    else:
        st.warning("Не удалось загрузить полную статистику игроков (возможно, сезон не начался).")
