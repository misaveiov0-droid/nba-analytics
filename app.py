import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import TeamDetails, CommonTeamRoster, PlayerCareerStats
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Настройка страницы ---
st.set_page_config(page_title="NBA Analytics Hub", layout="wide")
st.title("🏀 NBA Analytics: Статистика, Прогнозы и Вероятность")

# --- Боковая панель поиска ---
st.sidebar.header("🔍 Поиск команды")
all_teams = teams.get_teams()
team_names = {team['full_name']: team['id'] for team in all_teams}
selected_team_name = st.sidebar.selectbox("Выберите команду", team_names.keys())

if selected_team_name:
    team_id = team_names[selected_team_name]
    
    # --- 1. Загрузка данных о команде (исправленная часть) ---
    with st.spinner("Загрузка данных..."):
        try:
            # Получаем детальную информацию о команде через TeamDetails
            team_info = TeamDetails(team_id=team_id).get_data_frames()[0]
            # Получаем состав команды (работает без изменений)
            roster = CommonTeamRoster(team_id=team_id).get_data_frames()[0]
        except Exception as e:
            st.error(f"Ошибка API: {e}. Возможно, изменились заголовки запросов.")
            st.stop()

    # --- Интерфейс ---
    st.header(f"📊 {selected_team_name}")
    
    # Отображение ключевых метрик команды из TeamDetails
    col1, col2, col3, col4 = st.columns(4)
    if not team_info.empty:
        # Берем первую строку данных о команде
        team_row = team_info.iloc[0]
        # Пытаемся найти нужные колонки, если они есть
        wins = team_row.get('WINS', 'N/A')
        losses = team_row.get('LOSSES', 'N/A')
        win_pct = team_row.get('WIN_PCT', 'N/A')
        # Информацию об очках за игру в TeamDetails нет, поэтому добавим заглушку или оставим пустым
        col1.metric("🏆 Победы", wins)
        col2.metric("📉 Поражения", losses)
        if win_pct != 'N/A':
            col3.metric("⛹️ Win %", f"{float(win_pct):.1%}")
        col4.metric("📈 Забивают за игру", "—") # Заглушка
    else:
        st.warning("Не удалось загрузить общую статистику команды.")

    st.subheader("👥 Состав команды и персональные прогнозы")
    
    # --- 2. Аналитика по каждому игроку ---
    player_data = []
    # Ограничим количество игроков для скорости (например, первыми 10)
    for _, player in roster.head(10).iterrows():
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER']
        
        # Получаем карьерную статистику
        try:
            career = PlayerCareerStats(player_id=player_id).get_data_frames()[0]
            if len(career) >= 2:
                last_year = career.iloc[-1]
                prev_year = career.iloc[-2]
                
                # --- 3. Модель прогноза (ИИ) ---
                # Проверяем, что данные не пустые (иногда приходят NaN)
                if pd.notna(last_year['PTS']) and pd.notna(prev_year['PTS']):
                    X_train = [[prev_year['MIN'] or 0, prev_year['PTS'] or 0], 
                               [last_year['MIN'] or 0, last_year['PTS'] or 0]]
                    y_train = [prev_year['PTS'] or 0, last_year['PTS'] or 0]
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Прогноз
                    X_pred = [[last_year['MIN'] * 1.02 if last_year['MIN'] else 1, last_year['PTS'] or 0]]
                    predicted_pts = round(model.predict(X_pred)[0], 1)
                    
                    # Вероятность (чем стабильнее, тем выше)
                    volatility = abs(last_year['PTS'] - prev_year['PTS']) / (prev_year['PTS'] if prev_year['PTS'] else 1)
                    probability = max(0, min(100, round(100 - (volatility * 100), 0)))
                    
                    player_data.append({
                        "Имя": player_name,
                        "Позиция": player['POSITION'],
                        "PPG 2024": round(last_year['PTS'], 1),
                        "Прогноз PPG": predicted_pts,
                        "Вероятность (%)": probability,
                        "Подборы": last_year['REB'] or 0,
                        "Передачи": last_year['AST'] or 0
                    })
                else:
                    player_data.append({"Имя": player_name, "Позиция": player['POSITION'], "Статус": "Недостаточно данных"})
        except Exception as e:
            player_data.append({"Имя": player_name, "Позиция": player['POSITION'], "Статус": "Ошибка данных"})

    # Отображение таблицы игроков
    df_players = pd.DataFrame(player_data)
    if not df_players.empty and 'PPG 2024' in df_players.columns:
        st.dataframe(df_players, use_container_width=True)
        
        # График (только если есть прогнозы)
        if 'Прогноз PPG' in df_players.columns:
            fig = px.bar(df_players, x='Имя', y=['PPG 2024', 'Прогноз PPG'], 
                         title="Сравнение текущей результативности с ИИ-прогнозом",
                         barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Метрика вероятности
        if 'Вероятность (%)' in df_players.columns:
            avg_prob = df_players['Вероятность (%)'].mean()
            st.metric("🎲 Общая вероятность стабильности команды", f"{round(avg_prob)}%")
    else:
        st.warning("Не удалось загрузить полную статистику игроков.")
