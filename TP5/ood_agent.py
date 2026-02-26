import gymnasium as gym
from stable_baselines3 import PPO
from PIL import Image

print("--- ÉVALUATION OOD : GRAVITÉ FAIBLE ---")

# Création de l'environnement avec une gravité modifiée (doit être entre -12.0 et 0.0)
# Par exemple, utilisons -2.0 pour simuler une gravité lunaire réaliste
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=-2.0)

# Chargement du modèle entraîné à l'Exercice 2
# L'utilisation de device="cpu" est recommandée pour de l'inférence simple
model = PPO.load("ppo_lunar_lander", device="cpu")

obs, info = eval_env.reset()
done = False
frames = []

total_reward = 0.0
main_engine_uses = 0
side_engine_uses = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # Mise à jour des métriques
    total_reward += reward
    if action == 2:
        main_engine_uses += 1
    elif action in [1, 3]:
        side_engine_uses += 1
        
    frames.append(Image.fromarray(eval_env.render()))
    done = terminated or truncated

eval_env.close()

# Analyse du vol
if reward == -100:
    issue = "CRASH DÉTECTÉ 💥"
elif reward == 100:
    issue = "ATTERRISSAGE RÉUSSI 🏆"
else:
    issue = "TEMPS ÉCOULÉ OU SORTIE DE ZONE ⚠️"

print("\n--- RAPPORT DE VOL PPO (GRAVITÉ MODIFIÉE) ---")
print(f"Issue du vol : {issue}")
print(f"Récompense totale cumulée : {total_reward:.2f} points")
print(f"Allumages moteur principal : {main_engine_uses}")
print(f"Allumages moteurs latéraux : {side_engine_uses}")
print(f"Durée du vol : {len(frames)} frames")

if frames:
    frames[0].save('ood_agent.gif', save_all=True, append_images=frames[1:], duration=30, loop=0)
    print("Vidéo de la télémétrie sauvegardée sous 'ood_agent.gif'")
        