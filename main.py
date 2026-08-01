import time
import yaml
import logging
import os
from datetime import datetime
from database.db_manager import DatabaseManager
from scraper.scraper import Scraper
from environment.rl_env import BettingEnv
from agent.dqn_agent import DQNAgent

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

def load_config():
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    db = DatabaseManager(config['database']['path'])
    scraper = Scraper(config['scraper']['url'])
    env = BettingEnv(history_size=config['environment']['state_size'])
    
    # State dim: history_size + 7 (additional features)
    state_dim = config['environment']['state_size'] + 7
    # Actions: 0 (PRETO), 1 (VERDE)
    action_dim = 2
    
    agent = DQNAgent(state_dim, action_dim, config['agent'])
    
    logger.info("System started. Entering main loop...")
    
    consecutive_wins = 0
    last_state = None
    last_action = None
    
    # Use mock=True for initial testing if URL is not valid
    is_mock = config['scraper']['url'] == "https://example.com/double"
    if is_mock:
        logger.warning("Using MOCK mode for scraper because URL is default.")

    while True:
        try:
            # 1. Get current state from DB
            history_df = db.get_history(limit=config['environment']['state_size'])
            current_state = env.get_state(history_df)
            
            # 2. Agent Predicts
            action_idx = agent.select_action(current_state)
            predicted_color = "PRETO" if action_idx == 0 else "VERDE"
            
            logger.info(f"PREVISÃO: {predicted_color} (Epsilon: {agent.epsilon:.4f})")
            
            # 3. Wait for result (15s + small delay for validation)
            time.sleep(config['scraper']['interval'])
            
            # 4. Scrape actual result
            result = scraper.fetch_latest_result(mock=is_mock)
            if not result:
                logger.error("Failed to fetch result. Skipping iteration.")
                continue
                
            actual_color = result['color']
            db.save_result(actual_color, result['number'], result['date'], result['time'])
            
            # 5. Calculate Reward and Train
            is_correct = (predicted_color == actual_color)
            if is_correct:
                consecutive_wins += 1
            else:
                consecutive_wins = 0
                
            reward = env.calculate_reward(predicted_color, actual_color, consecutive_wins)
            db.save_prediction(predicted_color, actual_color, is_correct)
            
            # Store transition and update agent
            next_history_df = db.get_history(limit=config['environment']['state_size'])
            next_state = env.get_state(next_history_df)
            
            agent.store_transition(current_state, action_idx, reward, next_state, False)
            agent.update()
            
            # 6. Save agent periodically
            if agent.steps_done % config['metrics']['save_interval'] == 0:
                agent.save()
                logger.info("Model saved.")
            
            logger.info(f"RESULTADO: {actual_color} | ACERTO: {is_correct} | RECOMPENSA: {reward:.2f}")
            
        except KeyboardInterrupt:
            logger.info("System stopped by user.")
            agent.save()
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            time.sleep(5)

if __name__ == "__main__":
    main()
