import gymnasium as gym
from stable_baselines3 import PPO
from PIL import Image

print("--- PHASE 1 : ENTRAÎNEMENT ---")
# Environnement sans rendu visuel pour accélérer l'entraînement au maximum
train_env = gym.make("LunarLander-v3")

# Initialisation du modèle PPO avec un réseau de neurones multicouches classique (MLP)
# verbose=1 permet d'afficher les logs d'entraînement dans le terminal
model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")

# Lancement de l'apprentissage (500 000 itérations sont un bon point de départ)
model.learn(total_timesteps=500000)

# Sauvegarde du modèle sur le disque
model.save("ppo_lunar_lander")
train_env.close()
print("Entraînement terminé et modèle sauvegardé !")

print("\n--- PHASE 2 : ÉVALUATION ET TÉLÉMÉTRIE ---")
# Nouvel environnement avec le mode de rendu pour extraire les images
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Chargement du modèle (optionnel ici car il est déjà en mémoire, mais bonne pratique)
# model = PPO.load("ppo_lunar_lander")

obs, info = eval_env.reset()
done = False
frames = []

total_reward = 0.0
main_engine_uses = 0
side_engine_uses = 0

while not done:
    # L'agent PPO prédit la meilleure action à prendre. 
    # deterministic=True demande à l'agent de prendre la meilleure action connue, sans explorer.
    action, _states = model.predict(obs, deterministic=True)
    
    # L'action choisie est envoyée à l'environnement
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # Mise à jour des métriques
    total_reward += reward
    if action == 2:
        main_engine_uses += 1
    elif action in [1, 3]:
        side_engine_uses += 1
        
    # Capture de l'image
    frame = eval_env.render()
    frames.append(Image.fromarray(frame))
    
    done = terminated or truncated

eval_env.close()

# Analyse du vol
if reward == -100:
    issue = "CRASH DÉTECTÉ 💥"
elif reward == 100:
    issue = "ATTERRISSAGE RÉUSSI 🏆"
else:
    issue = "TEMPS ÉCOULÉ OU SORTIE DE ZONE ⚠️"

print("\n--- RAPPORT DE VOL PPO ---")
print(f"Issue du vol : {issue}")
print(f"Récompense totale cumulée : {total_reward:.2f} points")
print(f"Allumages moteur principal : {main_engine_uses}")
print(f"Allumages moteurs latéraux : {side_engine_uses}")
print(f"Durée du vol : {len(frames)} frames")

if frames:
    frames[0].save('trained_ppo_agent.gif', save_all=True, append_images=frames[1:], duration=30, loop=0)
    print("Vidéo de la télémétrie sauvegardée sous 'trained_ppo_agent.gif'")
        