"""
Setup and Run Orthanc PACS Server for Testing
==============================================
This script helps setup Orthanc (open-source PACS server) for testing DICOM workflows.
"""

import os
import sys
import subprocess
import requests
import time
import json
from pathlib import Path

class OrthancServer:
    def __init__(self, data_dir="orthanc_data"):
        self.data_dir = Path(data_dir)
        self.config_file = self.data_dir / "orthanc.json"
        self.http_port = 8042
        self.dicom_port = 4242
        self.process = None
        
    def create_config(self):
        """Create Orthanc configuration file"""
        self.data_dir.mkdir(exist_ok=True)
        
        config = {
            "Name": "RADIO_PROJET_PACS",
            "HttpPort": self.http_port,
            "DicomPort": self.dicom_port,
            "RemoteAccessAllowed": True,
            "AuthenticationEnabled": False,
            "RegisteredUsers": {
                "test": "test"
            },
            "DicomAet": "ORTHANC",
            "DicomCheckCalledAet": False,
            "DicomCheckModalityHost": False,
            "StorageDirectory": str(self.data_dir / "storage"),
            "IndexDirectory": str(self.data_dir / "index"),
            "StorageCompression": False,
            "MaximumStorageSize": 0,
            "MaximumPatientCount": 0,
            "Plugins": [],
            "ConcurrentJobs": 2,
            "HttpTimeout": 60,
            "DicomScpTimeout": 30,
            "UnknownSopClassAccepted": True,
            "SaveJobs": True,
            "JobsHistorySize": 1000,
            "SynchronousCMove": True,
            "DicomWeb": {
                "Enable": True,
                "Root": "/dicom-web/",
                "EnableWado": True,
                "WadoRoot": "/wado",
                "Ssl": False,
                "StowMaxInstances": 10,
                "StowMaxSize": 1024
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration créée: {self.config_file}")
        
    def check_installation(self):
        """Check if Orthanc is installed"""
        try:
            result = subprocess.run(["Orthanc", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Orthanc trouvé: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        print("✗ Orthanc n'est pas installé")
        return False
    
    def install_orthanc(self):
        """Instructions pour installer Orthanc"""
        print("\n" + "="*60)
        print("INSTALLATION D'ORTHANC")
        print("="*60)
        print("\nPour Windows:")
        print("1. Télécharger depuis: https://www.orthanc-server.com/download.php")
        print("2. Version recommandée: Orthanc 1.12.x (Windows 64-bit)")
        print("3. Extraire et ajouter au PATH système")
        print("\nAlternative avec Docker:")
        print("  docker run -p 4242:4242 -p 8042:8042 --rm jodogne/orthanc")
        print("\nPour Linux/Mac:")
        print("  sudo apt-get install orthanc  # Ubuntu/Debian")
        print("  brew install orthanc          # macOS")
        print("="*60)
        
    def start_server(self):
        """Start Orthanc server"""
        if not self.check_installation():
            self.install_orthanc()
            return False
        
        print(f"\n🚀 Démarrage d'Orthanc...")
        print(f"   HTTP: http://localhost:{self.http_port}")
        print(f"   DICOM: localhost:{self.dicom_port}")
        
        try:
            self.process = subprocess.Popen(
                ["Orthanc", str(self.config_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(3)
            
            if self.is_running():
                print("✓ Serveur Orthanc démarré avec succès!")
                return True
            else:
                print("✗ Échec du démarrage du serveur")
                return False
                
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return False
    
    def start_with_docker(self):
        """Start Orthanc using Docker"""
        print("\n🐳 Démarrage d'Orthanc avec Docker...")
        
        try:
            # Check if Docker is available
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("✗ Docker n'est pas installé")
                return False
            
            # Check if container already exists
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=orthanc-test", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            
            if "orthanc-test" in result.stdout:
                print("Container existant trouvé, redémarrage...")
                subprocess.run(["docker", "start", "orthanc-test"])
            else:
                print("Création d'un nouveau container...")
                subprocess.run([
                    "docker", "run", "-d",
                    "--name", "orthanc-test",
                    "-p", f"{self.dicom_port}:4242",
                    "-p", f"{self.http_port}:8042",
                    "-v", f"{self.data_dir.absolute()}:/var/lib/orthanc/db",
                    "jodogne/orthanc"
                ])
            
            time.sleep(5)
            
            if self.is_running():
                print("✓ Serveur Orthanc (Docker) démarré avec succès!")
                print(f"   Interface Web: http://localhost:{self.http_port}")
                print("   Credentials: orthanc / orthanc")
                return True
            else:
                print("✗ Échec du démarrage du container")
                return False
                
        except Exception as e:
            print(f"✗ Erreur Docker: {e}")
            return False
    
    def is_running(self):
        """Check if Orthanc server is running"""
        try:
            response = requests.get(f"http://localhost:{self.http_port}/system", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_info(self):
        """Get server information"""
        try:
            response = requests.get(f"http://localhost:{self.http_port}/system")
            if response.status_code == 200:
                info = response.json()
                print("\n" + "="*60)
                print("INFORMATIONS DU SERVEUR ORTHANC")
                print("="*60)
                print(f"Nom: {info.get('Name', 'N/A')}")
                print(f"Version: {info.get('Version', 'N/A')}")
                print(f"DICOM AET: {info.get('DicomAet', 'N/A')}")
                print(f"Storage: {info.get('StorageAreaPlugin', 'Default')}")
                print("="*60)
                return True
        except Exception as e:
            print(f"✗ Impossible d'obtenir les informations: {e}")
            return False
    
    def stop_server(self):
        """Stop Orthanc server"""
        if self.process:
            print("\n🛑 Arrêt du serveur...")
            self.process.terminate()
            self.process.wait()
            print("✓ Serveur arrêté")
        else:
            # Try stopping Docker container
            try:
                subprocess.run(["docker", "stop", "orthanc-test"], 
                             capture_output=True)
                print("✓ Container Docker arrêté")
            except:
                pass


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     SETUP ORTHANC PACS SERVER - RADIO_PROJET            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    server = OrthancServer()
    
    print("\nOptions:")
    print("1. Démarrer Orthanc (installation locale)")
    print("2. Démarrer Orthanc avec Docker (recommandé)")
    print("3. Créer uniquement la configuration")
    print("4. Vérifier l'état du serveur")
    print("5. Afficher les instructions d'installation")
    print("6. Quitter")
    
    choice = input("\nChoisissez une option (1-6): ").strip()
    
    if choice == "1":
        server.create_config()
        if server.start_server():
            server.get_info()
            print("\n💡 Le serveur reste actif. Fermez cette fenêtre pour l'arrêter.")
            try:
                server.process.wait()
            except KeyboardInterrupt:
                server.stop_server()
                
    elif choice == "2":
        if server.start_with_docker():
            server.get_info()
            print("\n💡 Container Docker en cours d'exécution.")
            print("   Pour l'arrêter: docker stop orthanc-test")
            print("   Pour le supprimer: docker rm orthanc-test")
            
    elif choice == "3":
        server.create_config()
        print(f"\n✓ Configuration créée dans: {server.config_file}")
        
    elif choice == "4":
        if server.is_running():
            print("✓ Serveur Orthanc est en cours d'exécution")
            server.get_info()
        else:
            print("✗ Serveur Orthanc n'est pas accessible")
            
    elif choice == "5":
        server.install_orthanc()
        
    else:
        print("Au revoir!")


if __name__ == "__main__":
    main()
