"""
Test PACS Connection and Upload DICOM Data
==========================================
Test connectivity with Orthanc PACS server and upload sample DICOM data
"""

import os
import sys
import requests
import pydicom
from pathlib import Path
from datetime import datetime
import json

class PACSConnectionTester:
    def __init__(self, server_url="http://localhost:8042", username=None, password=None):
        self.server_url = server_url.rstrip('/')
        self.auth = (username, password) if username and password else None
        self.session = requests.Session()
        if self.auth:
            self.session.auth = self.auth
    
    def test_connection(self):
        """Test basic connection to PACS server"""
        print("\n" + "="*60)
        print("TEST DE CONNEXION AU SERVEUR PACS")
        print("="*60)
        
        try:
            # Test system endpoint
            response = self.session.get(f"{self.server_url}/system", timeout=5)
            
            if response.status_code == 200:
                system_info = response.json()
                print("✓ Connexion réussie!")
                print(f"  Serveur: {system_info.get('Name', 'Unknown')}")
                print(f"  Version: {system_info.get('Version', 'Unknown')}")
                print(f"  DICOM AET: {system_info.get('DicomAet', 'Unknown')}")
                return True
            else:
                print(f"✗ Échec de connexion (Status: {response.status_code})")
                return False
                
        except requests.exceptions.ConnectionError:
            print("✗ Impossible de se connecter au serveur")
            print(f"  URL: {self.server_url}")
            print("\n💡 Assurez-vous que le serveur Orthanc est démarré.")
            print("   Lancez: python setup_orthanc_server.py")
            return False
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return False
    
    def get_statistics(self):
        """Get PACS statistics"""
        print("\n" + "="*60)
        print("STATISTIQUES DU SERVEUR")
        print("="*60)
        
        try:
            response = self.session.get(f"{self.server_url}/statistics")
            if response.status_code == 200:
                stats = response.json()
                print(f"Patients: {stats.get('CountPatients', 0)}")
                print(f"Études: {stats.get('CountStudies', 0)}")
                print(f"Séries: {stats.get('CountSeries', 0)}")
                print(f"Instances: {stats.get('CountInstances', 0)}")
                print(f"Espace disque: {stats.get('TotalDiskSizeMB', 0):.2f} MB")
                return stats
            else:
                print(f"✗ Impossible d'obtenir les statistiques (Status: {response.status_code})")
                return None
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return None
    
    def list_patients(self):
        """List all patients in PACS"""
        print("\n" + "="*60)
        print("LISTE DES PATIENTS")
        print("="*60)
        
        try:
            response = self.session.get(f"{self.server_url}/patients")
            if response.status_code == 200:
                patient_ids = response.json()
                
                if not patient_ids:
                    print("Aucun patient trouvé")
                    return []
                
                patients = []
                for patient_id in patient_ids[:10]:  # Limit to first 10
                    patient_response = self.session.get(f"{self.server_url}/patients/{patient_id}")
                    if patient_response.status_code == 200:
                        patient_data = patient_response.json()
                        main_info = patient_data.get('MainDicomTags', {})
                        patients.append({
                            'id': patient_id,
                            'name': main_info.get('PatientName', 'N/A'),
                            'patient_id': main_info.get('PatientID', 'N/A'),
                            'studies': len(patient_data.get('Studies', []))
                        })
                
                for i, patient in enumerate(patients, 1):
                    print(f"{i}. {patient['name']} (ID: {patient['patient_id']}) - {patient['studies']} études")
                
                if len(patient_ids) > 10:
                    print(f"\n... et {len(patient_ids) - 10} autres patients")
                
                return patients
            else:
                print(f"✗ Impossible de lister les patients (Status: {response.status_code})")
                return []
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return []
    
    def upload_dicom_file(self, file_path):
        """Upload a single DICOM file"""
        try:
            with open(file_path, 'rb') as f:
                dicom_data = f.read()
            
            response = self.session.post(
                f"{self.server_url}/instances",
                data=dicom_data,
                headers={'Content-Type': 'application/dicom'}
            )
            
            if response.status_code == 200:
                result = response.json()
                return True, result
            else:
                return False, f"Status: {response.status_code}"
                
        except Exception as e:
            return False, str(e)
    
    def upload_dicom_directory(self, directory_path, max_files=None):
        """Upload all DICOM files from a directory"""
        print("\n" + "="*60)
        print(f"UPLOAD DEPUIS: {directory_path}")
        print("="*60)
        
        directory = Path(directory_path)
        if not directory.exists():
            print(f"✗ Répertoire introuvable: {directory_path}")
            return 0, 0
        
        # Find all DICOM files
        dicom_files = []
        for ext in ['*.dcm', '*.DCM', '*.dicom']:
            dicom_files.extend(directory.rglob(ext))
        
        # Also check files without extension
        for file in directory.rglob('*'):
            if file.is_file() and file.suffix == '':
                try:
                    # Try to read as DICOM
                    pydicom.dcmread(str(file), stop_before_pixels=True)
                    dicom_files.append(file)
                except:
                    pass
        
        if not dicom_files:
            print("✗ Aucun fichier DICOM trouvé")
            return 0, 0
        
        # Limit files if specified
        if max_files and len(dicom_files) > max_files:
            print(f"⚠️  Limitation à {max_files} fichiers (sur {len(dicom_files)} trouvés)")
            dicom_files = dicom_files[:max_files]
        
        print(f"📁 {len(dicom_files)} fichiers DICOM trouvés")
        print("🔄 Upload en cours...")
        
        success_count = 0
        fail_count = 0
        
        for i, file_path in enumerate(dicom_files, 1):
            success, result = self.upload_dicom_file(file_path)
            
            if success:
                success_count += 1
                if i % 10 == 0 or i == len(dicom_files):
                    print(f"   Progress: {i}/{len(dicom_files)} fichiers uploadés")
            else:
                fail_count += 1
                if fail_count <= 5:  # Show first 5 errors
                    print(f"   ✗ Échec: {file_path.name} - {result}")
        
        print("\n" + "="*60)
        print(f"✓ Upload terminé: {success_count} réussis, {fail_count} échoués")
        print("="*60)
        
        return success_count, fail_count
    
    def test_query(self, patient_id=None):
        """Test DICOM query functionality"""
        print("\n" + "="*60)
        print("TEST DE REQUÊTE DICOM")
        print("="*60)
        
        try:
            if patient_id:
                # Query specific patient
                response = self.session.get(f"{self.server_url}/patients/{patient_id}")
            else:
                # Query all patients
                response = self.session.get(f"{self.server_url}/patients")
            
            if response.status_code == 200:
                print("✓ Requête DICOM réussie")
                return True
            else:
                print(f"✗ Échec de la requête (Status: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return False
    
    def download_study(self, study_id, output_dir):
        """Download a complete study as ZIP"""
        print(f"\n📥 Téléchargement de l'étude {study_id}...")
        
        try:
            response = self.session.get(f"{self.server_url}/studies/{study_id}/archive")
            
            if response.status_code == 200:
                output_path = Path(output_dir) / f"study_{study_id}.zip"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ Étude téléchargée: {output_path}")
                print(f"  Taille: {len(response.content) / (1024*1024):.2f} MB")
                return True
            else:
                print(f"✗ Échec du téléchargement (Status: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║      TEST CONNEXION PACS - RADIO_PROJET                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    server_url = input("URL du serveur PACS [http://localhost:8042]: ").strip() or "http://localhost:8042"
    
    use_auth = input("Utiliser l'authentification? (o/N): ").strip().lower() == 'o'
    username = None
    password = None
    
    if use_auth:
        username = input("Username: ").strip()
        password = input("Password: ").strip()
    
    tester = PACSConnectionTester(server_url, username, password)
    
    # Test connection
    if not tester.test_connection():
        print("\n❌ Impossible de continuer sans connexion au serveur")
        return
    
    # Show menu
    while True:
        print("\n" + "="*60)
        print("OPTIONS")
        print("="*60)
        print("1. Afficher les statistiques")
        print("2. Lister les patients")
        print("3. Upload un répertoire DICOM")
        print("4. Upload depuis DATA/NSCLC-Radiomics")
        print("5. Tester une requête DICOM")
        print("6. Télécharger une étude")
        print("7. Test complet (upload + query)")
        print("8. Quitter")
        
        choice = input("\nChoisissez une option (1-8): ").strip()
        
        if choice == "1":
            tester.get_statistics()
            
        elif choice == "2":
            tester.list_patients()
            
        elif choice == "3":
            directory = input("Chemin du répertoire DICOM: ").strip()
            max_files = input("Nombre max de fichiers (Enter pour tous): ").strip()
            max_files = int(max_files) if max_files.isdigit() else None
            tester.upload_dicom_directory(directory, max_files)
            
        elif choice == "4":
            data_dir = Path("DATA/NSCLC-Radiomics")
            if not data_dir.exists():
                data_dir = Path(input("Chemin vers NSCLC-Radiomics: ").strip())
            
            max_files = input("Nombre max de fichiers [100]: ").strip()
            max_files = int(max_files) if max_files.isdigit() else 100
            
            tester.upload_dicom_directory(data_dir, max_files)
            tester.get_statistics()
            
        elif choice == "5":
            tester.test_query()
            
        elif choice == "6":
            study_id = input("ID de l'étude: ").strip()
            output_dir = input("Répertoire de sortie [./downloads]: ").strip() or "./downloads"
            tester.download_study(study_id, output_dir)
            
        elif choice == "7":
            print("\n🔄 Exécution du test complet...")
            
            # Upload sample data
            data_dir = Path("DATA/NSCLC-Radiomics")
            if data_dir.exists():
                success, fail = tester.upload_dicom_directory(data_dir, max_files=50)
                
                if success > 0:
                    # Show statistics
                    tester.get_statistics()
                    
                    # Test query
                    tester.test_query()
                    
                    # List patients
                    tester.list_patients()
                    
                    print("\n✅ Test complet terminé avec succès!")
                else:
                    print("\n❌ Aucun fichier uploadé, impossible de continuer")
            else:
                print(f"\n❌ Répertoire de données introuvable: {data_dir}")
            
        elif choice == "8":
            print("\n👋 Au revoir!")
            break
        else:
            print("❌ Option invalide")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrompu par l'utilisateur")
