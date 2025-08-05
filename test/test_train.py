from environment import ChineseChessEnv
from dqn import DQNAgent


def test_training_loop():
    env = ChineseChessEnv()
    agent = DQNAgent(env, difficulty='medium')

    # 迷你训练测试
    for episode in range(3):  # 测试3个episode
        state = env.reset()
        done = False
        while not done:
            legal_moves = env.get_legal_moves()
            action = agent.select_action(state, legal_moves)
            next_state, reward, done, _ = env.step(action)

            # 测试经验存储
            agent.store_transition(state, action, reward, next_state, done)

            # 测试网络更新
            if len(agent.memory) > agent.batch_size:
                loss = agent.update()
                if loss is not None:
                    print(f"Episode {episode} | Loss: {loss:.4f}")

            state = next_state

        print(f"Episode {episode} 完成")


if __name__ == "__main__":
    test_training_loop()