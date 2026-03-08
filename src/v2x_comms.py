import json
import time
import paho.mqtt.client as mqtt

class V2XCommunicator:
    """
    Gère les communications V2X via MQTT pour le partage des objets détectés.
    """
    def __init__(self, vehicle_id: str, broker_ip: str = "127.0.0.1", broker_port: int = 1883):
        self.vehicle_id = vehicle_id
        self.broker_ip = broker_ip
        self.broker_port = broker_port
        
        # Initialisation du client MQTT
        self.client = mqtt.Client(client_id=f"roadeye_{self.vehicle_id}")
        
        # Configuration des callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        # Définition des topics
        self.pub_topic = f"roadeye/v2x/{self.vehicle_id}/perceptions"
        # Le '+' est un wildcard : on écoute tous les véhicules sur ce niveau d'arborescence
        self.sub_topic = "roadeye/v2x/+/perceptions" 
        
        # Buffer thread-safe (basique) pour stocker les objets reçus asynchrones
        self.received_objects: list[dict] = []

    def connect(self) -> None:
        """Connecte au broker et lance le thread réseau en arrière-plan."""
        try:
            self.client.connect(self.broker_ip, self.broker_port, keepalive=60)
            self.client.loop_start()  # Ne bloque pas le thread principal (vital pour la vidéo)
            print(f"[{self.vehicle_id}] Démarrage du client MQTT V2X.")
        except Exception as e:
            print(f"[{self.vehicle_id}] Erreur de connexion MQTT : {e}")

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            print(f"[{self.vehicle_id}] Connecté au broker MQTT avec succès.")
            # On s'abonne pour écouter les autres véhicules
            self.client.subscribe(self.sub_topic)
        else:
            print(f"[{self.vehicle_id}] Échec de connexion MQTT (code {rc}).")

    def _on_message(self, client, userdata, msg) -> None:
        """Callback déclenché à la réception d'un message sur un topic abonné."""
        # On ignore nos propres messages pour éviter les boucles
        if msg.topic == self.pub_topic:
            return

        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Optionnel : vérifier si les données sont trop vieilles (latence)
            # current_time = time.time()
            # if current_time - payload.get("timestamp", current_time) > 0.5:
            #     return 
            
            # On ajoute les objets reçus dans notre buffer
            for obj in payload.get("objects", []):
                self.received_objects.append(obj)
                
        except json.JSONDecodeError:
            print(f"[{self.vehicle_id}] Payload JSON invalide reçu sur {msg.topic}")

    def publish_perceptions(self, objects_list: list[dict]) -> None:
        """
        Publie la liste des objets détectés localement vers le reste du réseau.
        """
        if not objects_list:
            return
            
        payload = {
            "vehicle_id": self.vehicle_id,
            "timestamp": time.time(),
            "objects": objects_list
        }
        
        # QoS=0 (At most once) est généralement préféré en V2X pour privilégier 
        # la faible latence plutôt que la garantie de livraison d'une donnée qui sera obsolète 30ms plus tard.
        self.client.publish(self.pub_topic, json.dumps(payload), qos=0)

    def get_and_clear_v2x_objects(self) -> list[dict]:
        """
        Récupère les objets reçus depuis le dernier appel et vide le buffer.
        À appeler à chaque itération de la boucle principale de perception.
        """
        objects = self.received_objects.copy()
        self.received_objects.clear()
        return objects

    def disconnect(self) -> None:
        """Arrête proprement le client."""
        self.client.loop_stop()
        self.client.disconnect()
        print(f"[{self.vehicle_id}] Déconnecté du broker MQTT.")