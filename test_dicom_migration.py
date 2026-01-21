"""
Test DICOM Data Migration Workflow
==================================
Complete end-to-end test of DICOM migration from PACS to processing pipeline
"""

import os
import sys
import requests
import pydicom
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import json

class DICOMMigrationTester:
    def __init__(self, pacs_url="http://localhost:8042"):
        self.pacs_url = pacs_url.rstrip('/')
        self.test_dir = Path("test_migration_output")
        self.session = requests.Session()
        
    def setup_test_environment(self):
        """Create test directories"""
        print("\n🔧 Configuration de l'environnement de test...")
        
        # Create test directories
        self.test_dir.mkdir(exist_ok=True)
        (self.test_dir / "downloaded").mkdir(exist_ok=True)
        (self.test_dir / "processed").mkdir(exist_ok=True)
        (self.test_dir / "results").mkdir(exist_ok=True)
        
        print(f"✓ Répertoire de test créé: {self.test_dir}")
        
    def test_pacs_connection(self):
        """Test 1: PACS Connection"""
        print("\n" + "="*60)
        print("TEST 1: CONNEXION AU SERVEUR PACS")
        print("="*60)
        
        try:
            response = self.session.get(f"{self.pacs_url}/system", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print("✅ RÉUSSI - Connexion au serveur PACS")
                print(f"   Serveur: {info.get('Name')}")
                print(f"   Version: {info.get('Version')}")
                return True
            else:
                print(f"❌ ÉCHEC - Status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ ÉCHEC - {e}")
            print("\n💡 Démarrez le serveur PACS: python setup_orthanc_server.py")
            return False
    
    def test_patient_query(self):
        """Test 2: Query Patient Data"""
        print("\n" + "="*60)
        print("TEST 2: REQUÊTE DES DONNÉES PATIENTS")
        print("="*60)
        
        try:
            # Get list of patients
            response = self.session.get(f"{self.pacs_url}/patients")
            if response.status_code != 200:
                print("❌ ÉCHEC - Impossible de récupérer la liste des patients")
                return False, None
            
            patient_ids = response.json()
            if not patient_ids:
                print("⚠️  ATTENTION - Aucun patient trouvé dans le PACS")
                print("   Uploadez des données avec: python test_pacs_connection.py")
                return False, None
            
            # Get first patient details
            patient_id = patient_ids[0]
            response = self.session.get(f"{self.pacs_url}/patients/{patient_id}")
            
            if response.status_code == 200:
                patient_data = response.json()
                main_info = patient_data.get('MainDicomTags', {})
                
                print("✅ RÉUSSI - Données patient récupérées")
                print(f"   Patient ID: {main_info.get('PatientID', 'N/A')}")
                print(f"   Nom: {main_info.get('PatientName', 'N/A')}")
                print(f"   Études: {len(patient_data.get('Studies', []))}")
                
                return True, patient_data
            else:
                print(f"❌ ÉCHEC - Status: {response.status_code}")
                return False, None
                
        except Exception as e:
            print(f"❌ ÉCHEC - {e}")
            return False, None
    
    def test_download_study(self, patient_data):
        """Test 3: Download Study from PACS"""
        print("\n" + "="*60)
        print("TEST 3: TÉLÉCHARGEMENT D'UNE ÉTUDE")
        print("="*60)
        
        if not patient_data or not patient_data.get('Studies'):
            print("❌ ÉCHEC - Aucune étude disponible")
            return False, None
        
        try:
            study_id = patient_data['Studies'][0]
            
            # Download study as ZIP
            response = self.session.get(f"{self.pacs_url}/studies/{study_id}/archive")
            
            if response.status_code == 200:
                zip_path = self.test_dir / "downloaded" / f"study_{study_id}.zip"
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                # Extract ZIP
                import zipfile
                extract_dir = self.test_dir / "downloaded" / f"study_{study_id}"
                extract_dir.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Count DICOM files
                dicom_files = list(extract_dir.rglob('*.dcm'))
                if not dicom_files:
                    dicom_files = [f for f in extract_dir.rglob('*') if f.is_file()]
                
                print("✅ RÉUSSI - Étude téléchargée et extraite")
                print(f"   Fichiers DICOM: {len(dicom_files)}")
                print(f"   Taille: {len(response.content) / (1024*1024):.2f} MB")
                print(f"   Emplacement: {extract_dir}")
                
                return True, extract_dir
            else:
                print(f"❌ ÉCHEC - Status: {response.status_code}")
                return False, None
                
        except Exception as e:
            print(f"❌ ÉCHEC - {e}")
            return False, None
    
    def test_dicom_processing(self, study_dir):
        """Test 4: Process DICOM Files"""
        print("\n" + "="*60)
        print("TEST 4: TRAITEMENT DES FICHIERS DICOM")
        print("="*60)
        
        if not study_dir or not study_dir.exists():
            print("❌ ÉCHEC - Répertoire d'étude introuvable")
            return False
        
        try:
            # Find DICOM files
            dicom_files = list(study_dir.rglob('*.dcm'))
            if not dicom_files:
                dicom_files = [f for f in study_dir.rglob('*') if f.is_file()]
            
            if not dicom_files:
                print("❌ ÉCHEC - Aucun fichier DICOM trouvé")
                return False
            
            processed_count = 0
            error_count = 0
            
            print(f"📄 Traitement de {len(dicom_files)} fichiers...")
            
            for dicom_file in dicom_files[:10]:  # Process first 10 files
                try:
                    # Read DICOM
                    dcm = pydicom.dcmread(str(dicom_file))
                    
                    # Extract metadata
                    metadata = {
                        'PatientID': str(getattr(dcm, 'PatientID', 'Unknown')),
                        'StudyDate': str(getattr(dcm, 'StudyDate', 'Unknown')),
                        'Modality': str(getattr(dcm, 'Modality', 'Unknown')),
                        'SeriesDescription': str(getattr(dcm, 'SeriesDescription', 'Unknown'))
                    }
                    
                    # Check if it has image data
                    if hasattr(dcm, 'pixel_array'):
                        img_array = dcm.pixel_array
                        metadata['ImageShape'] = img_array.shape
                        metadata['ImageDtype'] = str(img_array.dtype)
                        metadata['ImageStats'] = {
                            'min': float(np.min(img_array)),
                            'max': float(np.max(img_array)),
                            'mean': float(np.mean(img_array))
                        }
                    
                    # Save metadata
                    output_file = self.test_dir / "processed" / f"{dicom_file.stem}_metadata.json"
                    with open(output_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 3:
                        print(f"   ⚠️  Erreur sur {dicom_file.name}: {e}")
            
            if processed_count > 0:
                print("✅ RÉUSSI - Fichiers DICOM traités")
                print(f"   Traités: {processed_count}")
                print(f"   Erreurs: {error_count}")
                print(f"   Métadonnées sauvegardées: {self.test_dir / 'processed'}")
                return True
            else:
                print("❌ ÉCHEC - Aucun fichier traité avec succès")
                return False
                
        except Exception as e:
            print(f"❌ ÉCHEC - {e}")
            return False
    
    def test_data_validation(self):
        """Test 5: Validate Processed Data"""
        print("\n" + "="*60)
        print("TEST 5: VALIDATION DES DONNÉES TRAITÉES")
        print("="*60)
        
        try:
            metadata_files = list((self.test_dir / "processed").glob("*_metadata.json"))
            
            if not metadata_files:
                print("❌ ÉCHEC - Aucune métadonnée trouvée")
                return False
            
            valid_count = 0
            total_images = 0
            modalities = set()
            
            for metadata_file in metadata_files:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Validate metadata
                if metadata.get('PatientID') and metadata.get('Modality'):
                    valid_count += 1
                    modalities.add(metadata.get('Modality'))
                    
                    if metadata.get('ImageShape'):
                        total_images += 1
            
            if valid_count > 0:
                print("✅ RÉUSSI - Données validées")
                print(f"   Métadonnées valides: {valid_count}/{len(metadata_files)}")
                print(f"   Images détectées: {total_images}")
                print(f"   Modalités: {', '.join(modalities)}")
                return True
            else:
                print("❌ ÉCHEC - Aucune donnée valide")
                return False
                
        except Exception as e:
            print(f"❌ ÉCHEC - {e}")
            return False
    
    def test_pipeline_integration(self):
        """Test 6: Integration with Existing Pipeline"""
        print("\n" + "="*60)
        print("TEST 6: INTÉGRATION AVEC LE PIPELINE EXISTANT")
        print("="*60)
        
        try:
            # Check if processing scripts exist
            required_scripts = [
                'extract_masks_from_rtstruct.py',
                'normalize_rtstruct_patients.py',
                'train_multi_organ.py'
            ]
            
            found_scripts = []
            missing_scripts = []
            
            for script in required_scripts:
                if Path(script).exists():
                    found_scripts.append(script)
                else:
                    missing_scripts.append(script)
            
            print(f"Scripts trouvés: {len(found_scripts)}/{len(required_scripts)}")
            for script in found_scripts:
                print(f"   ✓ {script}")
            
            for script in missing_scripts:
                print(f"   ✗ {script} (manquant)")
            
            # Try to import dataset module
            try:
                sys.path.insert(0, str(Path.cwd()))
                import dataset_multi_organ
                print("   ✓ Module dataset_multi_organ importé")
                has_dataset = True
            except ImportError as e:
                print(f"   ✗ Impossible d'importer dataset_multi_organ: {e}")
                has_dataset = False
            
            if len(found_scripts) >= 2 and has_dataset:
                print("✅ RÉUSSI - Le pipeline peut intégrer les données PACS")
                print("\n📋 Prochaines étapes pour l'intégration:")
                print("   1. Télécharger les données depuis PACS")
                print("   2. Extraire les masques RT-STRUCT")
                print("   3. Normaliser les patients")
                print("   4. Lancer l'entraînement")
                return True
            else:
                print("⚠️  ATTENTION - Intégration partielle possible")
                return False
                
        except Exception as e:
            print(f"❌ ÉCHEC - {e}")
            return False
    
    def generate_report(self, results):
        """Generate test report"""
        print("\n" + "="*70)
        print("RAPPORT DE TEST - MIGRATION DICOM")
        print("="*70)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        print(f"\nRésultats: {passed_tests}/{total_tests} tests réussis")
        print("\nDétail des tests:")
        
        for test_name, result in results.items():
            status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
            print(f"  {status} - {test_name}")
        
        # Overall status
        print("\n" + "="*70)
        if passed_tests == total_tests:
            print("🎉 SUCCÈS COMPLET - Tous les tests sont passés!")
            print("   La migration DICOM est prête pour la production.")
        elif passed_tests >= total_tests * 0.7:
            print("✅ SUCCÈS PARTIEL - La plupart des tests sont passés")
            print("   La migration DICOM fonctionne mais nécessite des ajustements.")
        else:
            print("❌ ÉCHEC - Plusieurs tests ont échoué")
            print("   La migration DICOM nécessite des corrections.")
        
        print("="*70)
        
        # Save report
        report_path = self.test_dir / "results" / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': passed_tests / total_tests,
                'results': {k: 'passed' if v else 'failed' for k, v in results.items()}
            }, f, indent=2)
        
        print(f"\n📄 Rapport sauvegardé: {report_path}")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         TEST COMPLET DE MIGRATION DICOM - RADIO_PROJET           ║
    ╚══════════════════════════════════════════════════════════════════╝
        """)
        
        # Setup
        self.setup_test_environment()
        
        # Run tests
        results = {}
        
        # Test 1: PACS Connection
        results['Connexion PACS'] = self.test_pacs_connection()
        if not results['Connexion PACS']:
            print("\n⚠️  Tests suivants ignorés (pas de connexion PACS)")
            self.generate_report(results)
            return
        
        # Test 2: Patient Query
        results['Requête Patient'], patient_data = self.test_patient_query()
        
        # Test 3: Download Study
        results['Téléchargement Étude'], study_dir = self.test_download_study(patient_data)
        
        # Test 4: DICOM Processing
        results['Traitement DICOM'] = self.test_dicom_processing(study_dir)
        
        # Test 5: Data Validation
        results['Validation Données'] = self.test_data_validation()
        
        # Test 6: Pipeline Integration
        results['Intégration Pipeline'] = self.test_pipeline_integration()
        
        # Generate report
        self.generate_report(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DICOM Migration Workflow')
    parser.add_argument('--pacs-url', default='http://localhost:8042',
                       help='URL du serveur PACS (default: http://localhost:8042)')
    parser.add_argument('--quick', action='store_true',
                       help='Exécuter seulement les tests de base')
    
    args = parser.parse_args()
    
    tester = DICOMMigrationTester(pacs_url=args.pacs_url)
    
    if args.quick:
        # Quick test - connection only
        tester.setup_test_environment()
        if tester.test_pacs_connection():
            print("\n✅ Test rapide réussi - Serveur PACS accessible")
        else:
            print("\n❌ Test rapide échoué - Vérifiez le serveur PACS")
    else:
        # Full test suite
        tester.run_all_tests()


if __name__ == "__main__":
    main()
