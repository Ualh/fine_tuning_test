import os
import requests
import urllib3


from transformers import AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Fonction pour afficher les modules linéaires et leurs noms
def print_linear_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name)

print_linear_modules(model)

# --- Patch global pour désactiver la vérification SSL ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request
# ---------------------------------------------------------

from huggingface_hub import login

# Exemple : récupère le token d'une variable d'environnement
HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("Le token HF_TOKEN doit être défini dans l'environnement")

# Login sans passer de paramètre 'session' ni 'verify'
login(token=HF_TOKEN)

print("Login réussi sans erreur SSL")


