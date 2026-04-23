import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import commonteamroster, playercareerstats, teamdetails
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Настройка страницы ---
st.set_page_config(page_title="NBA Analytics Hub", layout="wide")
st.title("🏀 NBA Analytics: Статистика, Прогнозы и Вероятность")

# --- Боковая панель поиска ---
st.sidebar.header("🔍 Поиск команды")

# Загружаем список команд
@st.cache_data
def load_teams():
    return teams.get_teams()

all_teams = load_teams()
team_names = {team['full_name']: team['id'] for team in all_teams}
selected_team_name = st.sidebar.selectbox("Выберите команду", list(team_names.keys()))

if selected_team_name:
    team_id = team_names[selected_team_name]
    
    # --- Загрузка данных ---
    with st.spinner("Загрузка данных о команде..."):
        try:
            # Получаем состав команды
            roster_response = commonteamroster.CommonTeamRoster(team_id=team_id)
            roster = roster_response.get_data_frames()[0]
            
            # Получаем детали команды
            team_details_response = teamdetails.TeamDetails(team_id=team_id)
            team_info = team_details_response.get_data_frames()[0]
            
            st.success(f"✅ Данные команды {selected_team_name} загружены!")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
            st.info("Попробуйте выбрать другую команду или обновить страницу")
            st.stop()
    
    # --- Отображение информации о команде ---
    st.header(f"📊 {selected_team_name}")
    
    # Показываем основную информацию
    col1, col2, col3 = st.columns(3)
    if not team_info.empty:
        team_row = team_info.iloc[0]
        col1.metric("🏆 Аббревиатура", team_row.get('TEAM_ABBREVIATION', '—'))
        col2.metric("📍 Город", team_row.get('TEAM_CITY', '—'))
        col3.metric("🏟️ Арена", team_row.get('ARENA_NAME', '—')[:20])
    
    st.subheader("👥 Состав команды")
    
    # --- Аналитика по игрокам ---
    player_stats = []
    progress_bar = st.progress(0)
    
    for idx, (_, player) in enumerate(roster.head(8).iterrows()):
        progress_bar.progress((idx + 1) / len(roster.head(8)))
        
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER']
        player_position = player.get('POSITION', 'N/A')
        
        try:
            # Получаем карьерную статистику игрока
            career_response = playercareerstats.PlayerCareerStats(player_id=player_id)
            career = career_response.get_data_frames()[0]
            
            if len(career) >= 2:
                # Берем последние 2 сезона
                last_season = career.iloc[-1]
                prev_season = career.iloc[-2]
                
                # Проверяем наличие данных
                if pd.notna(last_season['PTS']) and pd.notna(prev_season['PTS']):
                    # Простой прогноз на основе тренда
                    if last_season['PTS'] > prev_season['PTS']:
                        predicted_pts = round(last_season['PTS'] * 1.05, 1)  # +5% если прогресс
                    else:
                        predicted_pts = round(last_season['PTS'] * 0.95, 1)  # -5% если регресс
                    
                    # Вероятность улучшения
                    improvement = last_season['PTS'] - prev_season['PTS']
                    if improvement > 0:
                        probability = min(95, 50 + int(improvement * 5))
                    elif improvement < 0:
                        probability = max(5, 50 + int(improvement * 5))
                    else:
                        probability = 50
                    
                    player_stats.append({
                        "Игрок": player_name,
                        "Позиция": player_position,
                        "PPG (текущий)": round(last_season['PTS'], 1),
                        "PPG (прошлый)": round(prev_season['PTS'], 1),
                        "Прогноз PPG": predicted_pts,
                        "Вероятность роста": f"{probability}%",
                        "Минуты": round(last_season.get('MIN', 0), 1),
                        "Подборы": round(last_season.get('REB', 0), 1),
                        "Передачи": round(last_season.get('AST', 0), 1)
                    })
        except Exception as e:
            player_stats.append({
                "Игрок": player_name,
                "Позиция": player_position,
                "Статус": "Нет данных"
            })
    
    progress_bar.empty()
    
    # --- Отображение таблицы игроков ---
    if player_stats:
        df_players = pd.DataFrame(player_stats)
        
        # Убираем колонку со статусом если она есть
        if 'Статус' in df_players.columns:
            df_players = df_players[df_players['Статус'] != "Нет данных"]
        
        if not df_players.empty and 'PPG (текущий)' in df_players.columns:
            # ИСПРАВЛЕНО: use_container_width заменён на width='stretch'
            st.dataframe(df_players, width='stretch')
            
            # График сравнения
            fig = px.bar(
                df_players, 
                x='Игрок', 
                y=['PPG (текущий)', 'Прогноз PPG'],
                title="📈 Текущая результативность vs Прогноз ИИ",
                barmode='group',
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            # ИСПРАВЛЕНО: use_container_width заменён на width='stretch'
            st.plotly_chart(fig, width='stretch')
            
            # Дополнительная статистика
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_ppg = df_players['PPG (текущий)'].mean()
                st.metric("📊 Средний PPG команды", f"{round(avg_ppg, 1)}")
            with col2:
                top_scorer = df_players.loc[df_players['PPG (текущий)'].idxmax(), 'Игрок']
                st.metric("🏆 Лучший бомбардир", top_scorer)
            with col3:
                # Средняя вероятность (убираем знак % для подсчета)
                probabilities = [int(p.replace('%', '')) for p in df_players['Вероятность роста']]
                avg_prob = sum(probabilities) / len(probabilities)
                st.metric("🎲 Средняя вероятность роста", f"{round(avg_prob)}%")
            
            # Дополнительный график - распределение очков
            st.subheader("📊 Распределение очков по игрокам")
            fig2 = px.pie(
                df_players,
                values='PPG (текущий)',
                names='Игрок',
                title=f"Вклад игроков в атаку команды {selected_team_name}"
            )
            st.plotly_chart(fig2, width='stretch')
            
    else:
        st.warning("Не удалось загрузить статистику игроков")
    
    # Информация о версии API
    st.caption("Данные предоставлены NBA API. Прогнозы основаны на статистике последних сезонов.")
